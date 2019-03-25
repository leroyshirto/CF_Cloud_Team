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
            name='generated-model-dir', value='s3://output')):
    """A pipeline with two sequential steps."""

    fetch_step = dsl.ContainerOp(
        name='fetch',
        image='templum/openvino-fetch:2',
        command=['python3', 'app.py'],
        arguments=[],
        file_outputs={'downloaded': '/tmp/output_file'})

    prediction_step = dsl.ContainerOp(
        name='calc-engine',
        image='templum/openvino-serve:7',
        command=['python3', 'predict.py'],
        arguments=[
            '--model_bin', model_bin,
            '--model_xml', model_xml,
            '--input_numpy_file', fetch_step.output,
            '--output_bucket', generated_model_dir],
        file_outputs={'results': '/tmp/output'})

    submit_results_step = dsl.ContainerOp(
        name='submit-results',
        image='leroyshirtofh/oisp-results-submission:1',
        command=['python3', 'app.py'],
        arguments=['--results_json', prediction_step.output],
        file_outputs={})


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(emotion_pipeline, __file__ + '.tar.gz')
