"""CPU unit fixtures only. No benchmark records, GPU, Torch or Triton import."""
from types import SimpleNamespace
from capture_scope import CompileCapture
class FixtureJit:
 def run(self,fail=False):
  if fail:raise ValueError('CPU fixture exception')
  return kernel
class FixtureEntry:
 def run(self):return FixtureJit().run(),{'fixture':True}
kernel=SimpleNamespace(metadata={'fixture_only':True},asm={'ptx':'CPU_UNIT_FIXTURE_NEVER_ARCHIVED'})
jit_original,entry_original=FixtureJit.run,FixtureEntry.run
capture=CompileCapture()
for _ in range(2):
 capture.install(FixtureJit,FixtureEntry)
 try:FixtureEntry().run()
 finally:capture.restore()
 assert FixtureJit.run is jit_original and FixtureEntry.run is entry_original
 assert capture.observations==2 and len(capture.kernels)==1
 capture.assert_restored();capture.clear()
 FixtureEntry().run()  # stands for uninstrumented phase, no timing measurement
 assert capture.observations==0
capture.install(FixtureJit)
try:FixtureJit().run(fail=True)
except ValueError:pass
finally:capture.restore()
assert FixtureJit.run is jit_original;capture.assert_restored()
print('CPU unit PASS: restore before uninstrumented phase, fresh next case, LibEntry tuple handling, exception cleanup. No benchmark/GPU execution.')
