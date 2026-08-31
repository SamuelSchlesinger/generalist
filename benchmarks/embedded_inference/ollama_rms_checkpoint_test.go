package mlx

import (
	"math"
	"os"
	"testing"
)

func generalistCheckpointSummary(t *testing.T, name string, value *Array) {
	t.Helper()
	value32 := value.AsType(DTypeFloat32)
	Eval(value32)
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

// TestGeneralistRMSCheckpoint is copied into x/mlxrunner/mlx in a pinned
// Ollama v0.32.13 checkout. It intentionally uses Ollama's public wrappers for
// every operation so the differential covers its actual load/graph sequence.
func TestGeneralistRMSCheckpoint(t *testing.T) {
	weightPath := os.Getenv("GENERALIST_RMS_WEIGHT_PATH")
	weightName := os.Getenv("GENERALIST_RMS_WEIGHT_NAME")
	if weightPath == "" || weightName == "" {
		t.Skip("GENERALIST_RMS_WEIGHT_PATH and GENERALIST_RMS_WEIGHT_NAME are required")
	}

	skipIfNoMLX(t)
	withMLXThread(t, func() {
		const width = 5120
		values := make([]float32, width)
		for index := range values {
			centered := index%257 - 128
			fine := index%7 - 3
			values[index] = float32(centered)/64.0 + float32(fine)/1024.0
		}

		file, err := LoadSafetensorsNative(weightPath)
		if err != nil {
			t.Fatal(err)
		}
		defer file.Free()
		weight := file.Get(weightName)
		if weight == nil {
			t.Fatalf("missing weight %q", weightName)
		}

		input := FromValues(values, 1, 1, width).AsType(DTypeBFloat16)
		shiftedWeight := AddScalar(weight, 1.0)
		output := RMSNormFn(input, shiftedWeight, 1.0e-6)
		output32 := output.AsType(DTypeFloat32)
		Eval(output32)
		actual := output32.Floats()

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

		t.Logf("dtypes=%d/%d/%d width=%d", input.DType(), shiftedWeight.DType(), output.DType(), len(actual))
		t.Logf("sum=%.12f squared_sum=%.12f", sum, squaredSum)
		t.Logf("samples=%.9f/%.9f/%.9f", actual[0], actual[len(actual)/2], actual[len(actual)-1])
		t.Logf("fingerprint=%016x", fingerprint)
	})
}

func TestGeneralistEmbeddingCheckpoint(t *testing.T) {
	embeddingPath := os.Getenv("GENERALIST_EMBEDDING_WEIGHT_PATH")
	embeddingName := os.Getenv("GENERALIST_EMBEDDING_WEIGHT_NAME")
	normPath := os.Getenv("GENERALIST_RMS_WEIGHT_PATH")
	normName := os.Getenv("GENERALIST_RMS_WEIGHT_NAME")
	if embeddingPath == "" || embeddingName == "" || normPath == "" || normName == "" {
		t.Skip("embedding and RMSNorm checkpoint paths/names are required")
	}

	skipIfNoMLX(t)
	withMLXThread(t, func() {
		embeddingFile, err := LoadSafetensorsNative(embeddingPath)
		if err != nil {
			t.Fatal(err)
		}
		defer embeddingFile.Free()
		embeddingWeight := embeddingFile.Get(embeddingName)
		if embeddingWeight == nil {
			t.Fatalf("missing embedding weight %q", embeddingName)
		}

		normFile, err := LoadSafetensorsNative(normPath)
		if err != nil {
			t.Fatal(err)
		}
		defer normFile.Free()
		normWeight := normFile.Get(normName)
		if normWeight == nil {
			t.Fatalf("missing RMSNorm weight %q", normName)
		}

		indices := FromValues([]int32{42}, 1, 1)
		embedding := embeddingWeight.TakeAxis(indices, 0)
		normalized := RMSNormFn(embedding, AddScalar(normWeight, 1.0), 1.0e-6)
		generalistCheckpointSummary(t, "embedding", embedding)
		generalistCheckpointSummary(t, "normalized", normalized)
	})
}
