package qwen3_5

import (
	"math"
	"os"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func generalistFullLayerSummary(t *testing.T, name string, value *mlx.Array) {
	t.Helper()
	value32 := value.AsType(mlx.DTypeFloat32)
	mlx.Eval(value32)
	actual := value32.Floats()
	var sum float64
	var squaredSum float64
	fingerprint := uint64(1469598103934665603)
	for _, value := range actual {
		converted := float64(value)
		sum += converted
		squaredSum += converted * converted
		bits := math.Float32bits(value)
		for byteIndex := 0; byteIndex < 4; byteIndex++ {
			fingerprint ^= uint64(uint8(bits >> (8 * byteIndex)))
			fingerprint *= 1099511628211
		}
	}
	t.Logf("%s dtype=%d elements=%d", name, value.DType(), len(actual))
	t.Logf("%s sum=%.12f squared_sum=%.12f", name, sum, squaredSum)
	t.Logf("%s samples=%.9f/%.9f/%.9f", name, actual[0], actual[len(actual)/2], actual[len(actual)-1])
	t.Logf("%s fingerprint=%016x", name, fingerprint)
}

func TestGeneralistLayer0FullCheckpoint(t *testing.T) {
	paths := map[string]string{
		"embedding":  os.Getenv("GENERALIST_EMBEDDING_WEIGHT_PATH"),
		"input_norm": os.Getenv("GENERALIST_RMS_WEIGHT_PATH"),
		"qkv":        os.Getenv("GENERALIST_QKV_WEIGHT_PATH"),
		"z":          os.Getenv("GENERALIST_Z_WEIGHT_PATH"),
		"beta":       os.Getenv("GENERALIST_BETA_WEIGHT_PATH"),
		"alpha":      os.Getenv("GENERALIST_ALPHA_WEIGHT_PATH"),
		"conv":       os.Getenv("GENERALIST_CONV_WEIGHT_PATH"),
		"dt_bias":    os.Getenv("GENERALIST_DT_BIAS_PATH"),
		"a_log":      os.Getenv("GENERALIST_A_LOG_PATH"),
		"gdn_norm":   os.Getenv("GENERALIST_GDN_NORM_PATH"),
		"out_proj":   os.Getenv("GENERALIST_OUT_PROJ_PATH"),
		"post_norm":  os.Getenv("GENERALIST_POST_NORM_PATH"),
		"mlp_gate":   os.Getenv("GENERALIST_MLP_GATE_PATH"),
		"mlp_up":     os.Getenv("GENERALIST_MLP_UP_PATH"),
		"mlp_down":   os.Getenv("GENERALIST_MLP_DOWN_PATH"),
	}
	names := map[string]string{
		"embedding":  os.Getenv("GENERALIST_EMBEDDING_WEIGHT_NAME"),
		"input_norm": os.Getenv("GENERALIST_RMS_WEIGHT_NAME"),
		"qkv":        os.Getenv("GENERALIST_QKV_WEIGHT_NAME"),
		"z":          os.Getenv("GENERALIST_Z_WEIGHT_NAME"),
		"beta":       os.Getenv("GENERALIST_BETA_WEIGHT_NAME"),
		"alpha":      os.Getenv("GENERALIST_ALPHA_WEIGHT_NAME"),
		"conv":       os.Getenv("GENERALIST_CONV_WEIGHT_NAME"),
		"dt_bias":    os.Getenv("GENERALIST_DT_BIAS_NAME"),
		"a_log":      os.Getenv("GENERALIST_A_LOG_NAME"),
		"gdn_norm":   os.Getenv("GENERALIST_GDN_NORM_NAME"),
		"out_proj":   os.Getenv("GENERALIST_OUT_PROJ_NAME"),
		"post_norm":  os.Getenv("GENERALIST_POST_NORM_NAME"),
		"mlp_gate":   os.Getenv("GENERALIST_MLP_GATE_NAME"),
		"mlp_up":     os.Getenv("GENERALIST_MLP_UP_NAME"),
		"mlp_down":   os.Getenv("GENERALIST_MLP_DOWN_NAME"),
	}
	for key, path := range paths {
		if path == "" || names[key] == "" {
			t.Skip("all complete layer-zero checkpoint paths and names are required")
		}
	}

	mlxtest.Setup(t)
	files := make(map[string]*mlx.SafetensorsFile, len(paths))
	for key, path := range paths {
		file, err := mlx.LoadSafetensorsNative(path)
		if err != nil {
			t.Fatal(err)
		}
		files[key] = file
		defer file.Free()
	}
	get := func(key string) *mlx.Array {
		value := files[key].Get(names[key])
		if value == nil {
			t.Fatalf("checkpoint tensor %s is missing", key)
		}
		return value
	}
	quantized := func(key string) *nn.QuantizedLinear {
		linear := &nn.QuantizedLinear{
			Weight:      get(key),
			Scales:      files[key].Get(names[key] + ".scale"),
			GlobalScale: files[key].Get(names[key] + ".global_scale"),
			GroupSize:   16,
			Bits:        4,
			Mode:        "nvfp4",
		}
		if linear.Scales == nil || linear.GlobalScale == nil {
			t.Fatalf("checkpoint projection %s is missing NVFP4 scales", key)
		}
		return linear
	}

	packedQKVZ, packedBA, err := packGatedDeltaProjections(
		quantized("qkv"),
		quantized("z"),
		nn.NewLinear(get("beta"), nil),
		nn.NewLinear(get("alpha"), nil),
		nil,
		nil,
		&Config{},
	)
	if err != nil {
		t.Fatal(err)
	}
	indices := mlx.FromValues([]int32{42}, 1, 1)
	embedding := get("embedding").TakeAxis(indices, 0)
	normalized := mlx.RMSNormFn(embedding, mlx.AddScalar(get("input_norm"), 1.0), 1.0e-6)
	mixedQKVZ := packedQKVZ.Forward(normalized)
	mixedBA := packedBA.Forward(normalized)
	qkvInput := mlx.SliceStartStop(mixedQKVZ, []int32{0, 0, 0}, []int32{1, 1, 10240})
	zGate := mlx.SliceStartStop(mixedQKVZ, []int32{0, 0, 10240}, []int32{1, 1, 16384})
	zGate = mlx.Reshape(zGate, 1, 1, 48, 128)
	convWeight := mlx.Reshape(get("conv"), 10240, 4)
	priorConvState := mlx.Zeros(normalized.DType(), 1, 3, 10240)
	convWindow := mlx.Concatenate([]*mlx.Array{priorConvState, qkvInput}, 1)
	convOut := mlx.DepthwiseConvSiLU(convWindow, convWeight, nil, 1)
	aExp := mlx.Exp(get("a_log").AsType(mlx.DTypeFloat32))
	priorDeltaState := mlx.Zeros(mlx.DTypeFloat32, 1, 48, 128, 128)
	gdnOut, _, _ := mlx.GatedDelta(
		convOut,
		mixedBA,
		get("dt_bias"),
		aExp,
		priorDeltaState,
		nil,
		false,
	)
	gdnNorm := mlx.RMSNormFn(gdnOut, get("gdn_norm"), 1.0e-6)
	gated := mlx.Mul(
		gdnNorm.AsType(mlx.DTypeFloat32),
		mlx.SiLU(zGate.AsType(mlx.DTypeFloat32)),
	).AsType(gdnOut.DType())
	gated = mlx.Reshape(gated, 1, 1, 6144)
	attentionOut := quantized("out_proj").Forward(gated)
	afterAttention := mlx.Add(embedding, attentionOut)
	postNorm := mlx.RMSNormFn(
		afterAttention,
		mlx.AddScalar(get("post_norm"), 1.0),
		1.0e-6,
	)
	gateOut := quantized("mlp_gate").Forward(postNorm)
	upOut := quantized("mlp_up").Forward(postNorm)
	swiglu := mlx.SwiGLU(gateOut, upOut)
	mlpOut := quantized("mlp_down").Forward(swiglu)
	layerOut := mlx.Add(afterAttention, mlpOut)

	generalistFullLayerSummary(t, "gdn_norm", gdnNorm)
	generalistFullLayerSummary(t, "gated", gated)
	generalistFullLayerSummary(t, "attention_out", attentionOut)
	generalistFullLayerSummary(t, "after_attention", afterAttention)
	generalistFullLayerSummary(t, "post_norm", postNorm)
	generalistFullLayerSummary(t, "mlp_gate", gateOut)
	generalistFullLayerSummary(t, "mlp_up", upOut)
	generalistFullLayerSummary(t, "swiglu", swiglu)
	generalistFullLayerSummary(t, "mlp_out", mlpOut)
	generalistFullLayerSummary(t, "layer_out", layerOut)
}
