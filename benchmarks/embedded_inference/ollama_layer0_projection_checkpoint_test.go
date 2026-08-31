package qwen3_5

import (
	"math"
	"os"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func generalistLayer0Summary(t *testing.T, name string, value *mlx.Array) {
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

// TestGeneralistLayer0ProjectionCheckpoint exercises Ollama's actual packing
// helpers for Qwen's split layer-zero projections. It freezes the packed raw
// NVFP4 result, the post-global-scale QKVZ result, and the packed BF16
// beta/alpha projection before causal convolution or recurrent state exists.
func TestGeneralistLayer0ProjectionCheckpoint(t *testing.T) {
	embeddingPath := os.Getenv("GENERALIST_EMBEDDING_WEIGHT_PATH")
	embeddingName := os.Getenv("GENERALIST_EMBEDDING_WEIGHT_NAME")
	normPath := os.Getenv("GENERALIST_RMS_WEIGHT_PATH")
	normName := os.Getenv("GENERALIST_RMS_WEIGHT_NAME")
	qkvPath := os.Getenv("GENERALIST_QKV_WEIGHT_PATH")
	qkvName := os.Getenv("GENERALIST_QKV_WEIGHT_NAME")
	zPath := os.Getenv("GENERALIST_Z_WEIGHT_PATH")
	zName := os.Getenv("GENERALIST_Z_WEIGHT_NAME")
	betaPath := os.Getenv("GENERALIST_BETA_WEIGHT_PATH")
	betaName := os.Getenv("GENERALIST_BETA_WEIGHT_NAME")
	alphaPath := os.Getenv("GENERALIST_ALPHA_WEIGHT_PATH")
	alphaName := os.Getenv("GENERALIST_ALPHA_WEIGHT_NAME")
	if embeddingPath == "" || embeddingName == "" || normPath == "" || normName == "" ||
		qkvPath == "" || qkvName == "" || zPath == "" || zName == "" ||
		betaPath == "" || betaName == "" || alphaPath == "" || alphaName == "" {
		t.Skip("embedding, norm, QKV, Z, beta, and alpha checkpoint paths/names are required")
	}

	mlxtest.Setup(t)
	embeddingFile, err := mlx.LoadSafetensorsNative(embeddingPath)
	if err != nil {
		t.Fatal(err)
	}
	defer embeddingFile.Free()
	normFile, err := mlx.LoadSafetensorsNative(normPath)
	if err != nil {
		t.Fatal(err)
	}
	defer normFile.Free()
	qkvFile, err := mlx.LoadSafetensorsNative(qkvPath)
	if err != nil {
		t.Fatal(err)
	}
	defer qkvFile.Free()
	zFile, err := mlx.LoadSafetensorsNative(zPath)
	if err != nil {
		t.Fatal(err)
	}
	defer zFile.Free()
	betaFile, err := mlx.LoadSafetensorsNative(betaPath)
	if err != nil {
		t.Fatal(err)
	}
	defer betaFile.Free()
	alphaFile, err := mlx.LoadSafetensorsNative(alphaPath)
	if err != nil {
		t.Fatal(err)
	}
	defer alphaFile.Free()

	embeddingWeight := embeddingFile.Get(embeddingName)
	normWeight := normFile.Get(normName)
	qkv := &nn.QuantizedLinear{
		Weight:      qkvFile.Get(qkvName),
		Scales:      qkvFile.Get(qkvName + ".scale"),
		GlobalScale: qkvFile.Get(qkvName + ".global_scale"),
		GroupSize:   16,
		Bits:        4,
		Mode:        "nvfp4",
	}
	z := &nn.QuantizedLinear{
		Weight:      zFile.Get(zName),
		Scales:      zFile.Get(zName + ".scale"),
		GlobalScale: zFile.Get(zName + ".global_scale"),
		GroupSize:   16,
		Bits:        4,
		Mode:        "nvfp4",
	}
	betaWeight := betaFile.Get(betaName)
	alphaWeight := alphaFile.Get(alphaName)
	if embeddingWeight == nil || normWeight == nil || qkv.Weight == nil || qkv.Scales == nil ||
		qkv.GlobalScale == nil || z.Weight == nil || z.Scales == nil || z.GlobalScale == nil ||
		betaWeight == nil || alphaWeight == nil {
		t.Fatal("checkpoint is missing one or more required arrays")
	}

	packedQKVZLayer, packedBALayer, err := packGatedDeltaProjections(
		qkv,
		z,
		nn.NewLinear(betaWeight, nil),
		nn.NewLinear(alphaWeight, nil),
		nil,
		nil,
		&Config{},
	)
	if err != nil {
		t.Fatal(err)
	}
	packedQKVZ, ok := packedQKVZLayer.(*nn.QuantizedLinear)
	if !ok {
		t.Fatalf("packed QKVZ has unexpected type %T", packedQKVZLayer)
	}
	if _, ok := packedBALayer.(*nn.Linear); !ok {
		t.Fatalf("packed BA has unexpected type %T", packedBALayer)
	}

	indices := mlx.FromValues([]int32{42}, 1, 1)
	embedding := embeddingWeight.TakeAxis(indices, 0)
	normalized := mlx.RMSNormFn(embedding, mlx.AddScalar(normWeight, 1.0), 1.0e-6)
	rawQKVZ := mlx.QuantizedMatmul(
		normalized,
		packedQKVZ.Weight,
		packedQKVZ.Scales,
		packedQKVZ.QBiases,
		true,
		packedQKVZ.GroupSize,
		packedQKVZ.Bits,
		packedQKVZ.Mode,
	)
	mixedQKVZ := packedQKVZLayer.Forward(normalized)
	mixedBA := packedBALayer.Forward(normalized)

	mlx.Eval(qkv.GlobalScale, z.GlobalScale)
	t.Logf("widths=5120/10240/6144/96 global_scales=%.12f/%.12f", qkv.GlobalScale.Floats()[0], z.GlobalScale.Floats()[0])
	generalistLayer0Summary(t, "normalized_input", normalized)
	generalistLayer0Summary(t, "raw_qkvz", rawQKVZ)
	generalistLayer0Summary(t, "mixed_qkvz", mixedQKVZ)
	generalistLayer0Summary(t, "mixed_ba", mixedBA)
}
