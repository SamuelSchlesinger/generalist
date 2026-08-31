package qwen3_5

import (
	"math"
	"os"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func generalistGDNSummary(t *testing.T, name string, value *mlx.Array) {
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

func TestGeneralistLayer0GDNCheckpoint(t *testing.T) {
	paths := map[string]string{
		"embedding": os.Getenv("GENERALIST_EMBEDDING_WEIGHT_PATH"),
		"norm":      os.Getenv("GENERALIST_RMS_WEIGHT_PATH"),
		"qkv":       os.Getenv("GENERALIST_QKV_WEIGHT_PATH"),
		"z":         os.Getenv("GENERALIST_Z_WEIGHT_PATH"),
		"beta":      os.Getenv("GENERALIST_BETA_WEIGHT_PATH"),
		"alpha":     os.Getenv("GENERALIST_ALPHA_WEIGHT_PATH"),
		"conv":      os.Getenv("GENERALIST_CONV_WEIGHT_PATH"),
		"dt_bias":   os.Getenv("GENERALIST_DT_BIAS_PATH"),
		"a_log":     os.Getenv("GENERALIST_A_LOG_PATH"),
	}
	names := map[string]string{
		"embedding": os.Getenv("GENERALIST_EMBEDDING_WEIGHT_NAME"),
		"norm":      os.Getenv("GENERALIST_RMS_WEIGHT_NAME"),
		"qkv":       os.Getenv("GENERALIST_QKV_WEIGHT_NAME"),
		"z":         os.Getenv("GENERALIST_Z_WEIGHT_NAME"),
		"beta":      os.Getenv("GENERALIST_BETA_WEIGHT_NAME"),
		"alpha":     os.Getenv("GENERALIST_ALPHA_WEIGHT_NAME"),
		"conv":      os.Getenv("GENERALIST_CONV_WEIGHT_NAME"),
		"dt_bias":   os.Getenv("GENERALIST_DT_BIAS_NAME"),
		"a_log":     os.Getenv("GENERALIST_A_LOG_NAME"),
	}
	for key, path := range paths {
		if path == "" || names[key] == "" {
			t.Skip("all layer-zero GDN checkpoint paths and names are required")
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

	qkvWeight := get("qkv")
	zWeight := get("z")
	qkv := &nn.QuantizedLinear{
		Weight:      qkvWeight,
		Scales:      files["qkv"].Get(names["qkv"] + ".scale"),
		GlobalScale: files["qkv"].Get(names["qkv"] + ".global_scale"),
		GroupSize:   16,
		Bits:        4,
		Mode:        "nvfp4",
	}
	z := &nn.QuantizedLinear{
		Weight:      zWeight,
		Scales:      files["z"].Get(names["z"] + ".scale"),
		GlobalScale: files["z"].Get(names["z"] + ".global_scale"),
		GroupSize:   16,
		Bits:        4,
		Mode:        "nvfp4",
	}
	if qkv.Scales == nil || qkv.GlobalScale == nil || z.Scales == nil || z.GlobalScale == nil {
		t.Fatal("checkpoint is missing NVFP4 scales")
	}
	packedQKVZ, packedBA, err := packGatedDeltaProjections(
		qkv,
		z,
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
	normalized := mlx.RMSNormFn(embedding, mlx.AddScalar(get("norm"), 1.0), 1.0e-6)
	mixedQKVZ := packedQKVZ.Forward(normalized)
	mixedBA := packedBA.Forward(normalized)
	qkvInput := mlx.SliceStartStop(mixedQKVZ, []int32{0, 0, 0}, []int32{1, 1, 10240})
	convWeight := mlx.Reshape(get("conv"), 10240, 4)
	priorConvState := mlx.Zeros(normalized.DType(), 1, 3, 10240)
	convWindow := mlx.Concatenate([]*mlx.Array{priorConvState, qkvInput}, 1)
	convOut := mlx.DepthwiseConvSiLU(convWindow, convWeight, nil, 1)
	convState := mlx.SliceStartStop(convWindow, []int32{0, 1, 0}, []int32{1, 4, 10240})
	aExp := mlx.Exp(get("a_log").AsType(mlx.DTypeFloat32))
	priorDeltaState := mlx.Zeros(mlx.DTypeFloat32, 1, 48, 128, 128)
	gdnOut, deltaState, _ := mlx.GatedDelta(
		convOut,
		mixedBA,
		get("dt_bias"),
		aExp,
		priorDeltaState,
		nil,
		false,
	)

	generalistGDNSummary(t, "conv_out", convOut)
	generalistGDNSummary(t, "conv_state", convState)
	generalistGDNSummary(t, "a_exp", aExp)
	generalistGDNSummary(t, "gdn_out", gdnOut)
	generalistGDNSummary(t, "delta_state", deltaState)
}
