package qwen3_5

import (
	"math"
	"os"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func generalistNVFP4Summary(t *testing.T, name string, value *mlx.Array) {
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

// TestGeneralistNVFP4ProjectionCheckpoint exercises the real Ollama wrapper
// sequence for the first compressed operation in Qwen layer zero. The raw
// projection isolates MLX quantized_matmul; output additionally covers
// Ollama's compiled F32 global-scale multiply and BF16 cast.
func TestGeneralistNVFP4ProjectionCheckpoint(t *testing.T) {
	embeddingPath := os.Getenv("GENERALIST_EMBEDDING_WEIGHT_PATH")
	embeddingName := os.Getenv("GENERALIST_EMBEDDING_WEIGHT_NAME")
	normPath := os.Getenv("GENERALIST_RMS_WEIGHT_PATH")
	normName := os.Getenv("GENERALIST_RMS_WEIGHT_NAME")
	projectionPath := os.Getenv("GENERALIST_NVFP4_WEIGHT_PATH")
	weightName := os.Getenv("GENERALIST_NVFP4_WEIGHT_NAME")
	scaleName := os.Getenv("GENERALIST_NVFP4_SCALE_NAME")
	globalScaleName := os.Getenv("GENERALIST_NVFP4_GLOBAL_SCALE_NAME")
	if embeddingPath == "" || embeddingName == "" || normPath == "" || normName == "" ||
		projectionPath == "" || weightName == "" || scaleName == "" || globalScaleName == "" {
		t.Skip("embedding, norm, and NVFP4 checkpoint paths/names are required")
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
	projectionFile, err := mlx.LoadSafetensorsNative(projectionPath)
	if err != nil {
		t.Fatal(err)
	}
	defer projectionFile.Free()

	embeddingWeight := embeddingFile.Get(embeddingName)
	normWeight := normFile.Get(normName)
	weight := projectionFile.Get(weightName)
	scales := projectionFile.Get(scaleName)
	globalScale := projectionFile.Get(globalScaleName)
	if embeddingWeight == nil || normWeight == nil || weight == nil || scales == nil || globalScale == nil {
		t.Fatal("checkpoint is missing one or more required arrays")
	}

	indices := mlx.FromValues([]int32{42}, 1, 1)
	embedding := embeddingWeight.TakeAxis(indices, 0)
	normalized := mlx.RMSNormFn(embedding, mlx.AddScalar(normWeight, 1.0), 1.0e-6)
	raw := mlx.QuantizedMatmul(normalized, weight, scales, nil, true, 16, 4, "nvfp4")
	linear := &nn.QuantizedLinear{
		Weight:      weight,
		Scales:      scales,
		GlobalScale: globalScale,
		GroupSize:   16,
		Bits:        4,
		Mode:        "nvfp4",
	}
	output := linear.Forward(normalized)

	mlx.Eval(globalScale)
	t.Logf("layout=16/4 dtypes=%d/%d/%d global_scale=%.12f", weight.DType(), scales.DType(), globalScale.DType(), globalScale.Floats()[0])
	generalistNVFP4Summary(t, "normalized_input", normalized)
	generalistNVFP4Summary(t, "raw_projection", raw)
	generalistNVFP4Summary(t, "output", output)
}
