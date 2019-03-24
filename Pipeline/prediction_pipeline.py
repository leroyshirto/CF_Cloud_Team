import kfp.dsl as dsl


@dsl.pipeline(
    name='Prediction pipeline',
    description='Execute prediction operation for the dataset from numpy file and test accuracy and latency'
)
def emotion_pipeline(
        model_bin=dsl.PipelineParam(
            name='model-bin-path', value='s3://models/emotions_f32.bin'),
        model_xml=dsl.PipelineParam(
            name='model-xml-path', value='s3://models/emotions_f32.xml'),
        generated_model_dir=dsl.PipelineParam(
            name='generated-model-dir', value='s3://output'),
        input_numpy_file=dsl.PipelineParam(
            name='input-numpy-file', value='s3://generated-models/imgs.npy'),
        label_numpy_file=dsl.PipelineParam(
            name='label-numpy-file', value='s3://generated-models/lbs.npy'),
        batch_size=dsl.PipelineParam(name='batch-size', value=1),
        scale_div=dsl.PipelineParam(name='scale-input-divide', value=1),
        scale_sub=dsl.PipelineParam(name='scale-input-substract', value=0)):
    """A pipeline with two sequential steps."""

    fetch_step = dsl.ContainerOp(
        name='fetch',
        image='templum/openvino-fetch',
        command=['python3', 'app.py'],
        arguments=[],
        file_outputs={'downloaded': '/tmp/output_file'})

    build_step = dsl.ContainerOp(
        name='calc-engine',
        image='templum/openvino-serve:3',
        command=['python3', 'predict.py'],
        arguments=[
            '--model_bin', model_bin,
            '--model_xml', model_xml,
            '--input_numpy_file', fetch_step.output,
            '--label_numpy_file', label_numpy_file,
            '--batch_size', batch_size,
            '--scale_div', scale_div,
            '--scale_sub', scale_sub,
            '--output_bucket', generated_model_dir],
        file_outputs={})


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(emotion_pipeline, __file__ + '.tar.gz')
