package mlxrunner

import (
	"math"
	"os"
	"testing"
	"time"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/qwen3_5"
)

func generalistCachedDecodeSummary(t *testing.T, name string, value *mlx.Array) {
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

func TestGeneralistCachedDecodeCheckpoint(t *testing.T) {
	modelName := os.Getenv("GENERALIST_MODEL")
	if modelName == "" {
		t.Skip("GENERALIST_MODEL is required")
	}

	mlxtest.Setup(t)
	runner := &Runner{}
	if err := runner.Load(modelName); err != nil {
		t.Fatal(err)
	}
	model, ok := runner.Model.(*qwen3_5.Model)
	if !ok {
		t.Fatalf("model = %T, want *qwen3_5.Model", runner.Model)
	}
	t.Logf("rope_theta=%g rope_dim=%d attention_scale=%g", model.RopeTheta, model.RopeDim, model.Scale)
	caches := runner.Model.NewCaches()

	decode := func(offset int32, token int32) (*mlx.Array, *mlx.Array, int32) {
		input := mlx.FromValues([]int32{token}, 1, 1)
		b := &batch.Batch{
			InputIDs:     input,
			SeqOffsets:   []int32{offset},
			SeqQueryLens: []int32{1},
		}
		hidden, _ := runner.Model.Forward(b, caches)
		logits := runner.Model.Unembed(hidden)
		top := logits.Argmax(-1, false)
		mlx.Eval(top)
		return hidden, logits, int32(top.Int())
	}

	step0Started := time.Now()
	hidden0, logits0, token1 := decode(0, 42)
	t.Logf("step0 input=42 argmax=%d", token1)
	generalistCachedDecodeSummary(t, "step0_hidden", hidden0)
	generalistCachedDecodeSummary(t, "step0_logits", logits0)
	t.Logf("step0_complete_milliseconds=%.3f", float64(time.Since(step0Started).Microseconds())/1000)

	step1Started := time.Now()
	hidden1, logits1, token2 := decode(1, token1)
	t.Logf("step1 input=%d argmax=%d", token1, token2)
	generalistCachedDecodeSummary(t, "step1_hidden", hidden1)
	generalistCachedDecodeSummary(t, "step1_logits", logits1)
	t.Logf("step1_complete_milliseconds=%.3f", float64(time.Since(step1Started).Microseconds())/1000)
}
