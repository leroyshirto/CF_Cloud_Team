import kfp.dsl as dsl


@dsl.pipeline(
  name='Sequential',
  description='A pipeline with two sequential steps.'
)
def sequential_pipeline():
  """A pipeline with two sequential steps."""

  fetch_step = dsl.ContainerOp(
     name='download',
     image='library/bash:4.4.23',
     command=['sh', '-c'],
     arguments=['touch /tmp/test-output'],
     file_outputs={'downloaded': '/tmp/test-output'})

  inference = dsl.ContainerOp(
     name='echo',
     image='library/bash:4.4.23',
     command=['sh', '-c'],
     arguments=['echo "%s"' % fetch_step.output])

if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(sequential_pipeline, __file__ + '.tar.gz')