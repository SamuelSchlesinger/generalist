use generalist::provider::embedded::{MlxQwenModel, MlxRuntime, ModelPack};
use std::env;
use std::error::Error;
use std::path::PathBuf;

#[derive(Clone, Copy)]
enum TracePath {
    Baseline,
    ExactScale,
}

impl TracePath {
    fn parse(value: Option<std::ffi::OsString>) -> Result<Self, Box<dyn Error>> {
        match value {
            None => Ok(Self::ExactScale),
            Some(value) if value == "baseline" => Ok(Self::Baseline),
            Some(value) if value == "scale" => Ok(Self::ExactScale),
            Some(_) => Err("PATH must be baseline or scale".into()),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::ExactScale => "scale",
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let metallib = required_path(arguments.next(), "METALLIB")?;
    let manifest = required_path(arguments.next(), "MANIFEST")?;
    let blob_root = required_path(arguments.next(), "BLOB_ROOT")?;
    let steps = match arguments.next() {
        Some(value) => value
            .into_string()
            .map_err(|_| "STEPS must be valid UTF-8")?
            .parse::<usize>()?,
        None => 8,
    };
    let path = TracePath::parse(arguments.next())?;
    if arguments.next().is_some() || !(1..=100).contains(&steps) {
        return Err(
            "usage: embedded_decode_trace METALLIB MANIFEST BLOB_ROOT [STEPS=8] [PATH=baseline|scale]"
                .into(),
        );
    }

    let pack = ModelPack::open_ollama(manifest, blob_root)?;
    let runtime = MlxRuntime::initialize(metallib)?;
    let (mut model, load) = MlxQwenModel::load(&runtime, &pack)?;
    let mut session = model.start_session()?;
    let seed = match path {
        TracePath::Baseline => model.cached_decode_unfused_scale_checkpoint(&mut session, 42)?,
        TracePath::ExactScale => model.cached_decode_checkpoint(&mut session, 42)?,
    };
    if seed.position != 0
        || seed.decode.output_token_id != 1219
        || seed.decode.hidden.fingerprint != 0x6dfc_18bc_2b1c_2044
        || seed.decode.logits.fingerprint != 0x4efe_fa79_3523_b51f
    {
        return Err("trace seed diverged from the exact position-zero oracle".into());
    }

    let mut input = seed.decode.output_token_id;
    let mut output = Vec::with_capacity(steps);
    for index in 0..steps {
        let step = match path {
            TracePath::Baseline => model.cached_decode_argmax_unfused_scale(&mut session, input)?,
            TracePath::ExactScale => model.cached_decode_argmax(&mut session, input)?,
        };
        if step.position != index + 1 || step.input_token_id != input {
            return Err("trace decode session position diverged".into());
        }
        input = step.output_token_id;
        output.push(step.output_token_id);
    }
    if output.first() != Some(&73) {
        return Err("trace decode diverged from the position-one oracle".into());
    }

    println!(
        "path={} model_blobs={} arrays={} layers={}/{}/{} traced_steps={}",
        path.name(),
        load.blob_count,
        load.array_count,
        load.layer_count,
        load.linear_layer_count,
        load.full_attention_layer_count,
        steps
    );
    println!(
        "output_tokens={}",
        output
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(())
}

fn required_path(value: Option<std::ffi::OsString>, name: &str) -> Result<PathBuf, Box<dyn Error>> {
    value
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing {name} argument").into())
}
