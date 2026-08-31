package mlxrunner

import (
	"math"
	"os"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func generalistDecodeSummary(t *testing.T, name string, value *mlx.Array) {
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
	t.Logf(
		"%s samples=%.9f/%.9f/%.9f",
		name,
		actual[0],
		actual[len(actual)/2],
		actual[len(actual)-1],
	)
	t.Logf("%s fingerprint=%016x", name, fingerprint)
}

func TestGeneralistFullDecodeCheckpoint(t *testing.T) {
	modelName := os.Getenv("GENERALIST_MODEL")
	if modelName == "" {
		t.Skip("GENERALIST_MODEL is required")
	}

	mlxtest.Setup(t)
	runner := &Runner{}
	if err := runner.Load(modelName); err != nil {
		t.Fatal(err)
	}
	input := mlx.FromValues([]int32{42}, 1, 1)
	b := &batch.Batch{
		InputIDs:     input,
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{1},
	}
	// Nil caches make this the exact first-token, zero-state reference used by
	// Generalist. Each full-attention layer has one causal key, so its softmax
	// is exactly one regardless of Q/K normalization and position-zero RoPE.
	hidden, _ := runner.Model.Forward(b, nil)
	logits := runner.Model.Unembed(hidden)
	top := logits.Argmax(-1, false)
	mlx.Eval(top)
	t.Logf("argmax=%d", top.Int())
	generalistDecodeSummary(t, "hidden", hidden)
	generalistDecodeSummary(t, "logits", logits)
}
