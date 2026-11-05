import functools,inspect,json,os,pathlib,re,subprocess,sys,time
_ORIGINAL=subprocess.Popen
_SIGNATURE=inspect.signature(_ORIGINAL.__init__)
_COMPILERS=frozenset(['/opt/dtk/aillvm/bin/clang-22','/opt/dtk/aillvm/bin/clang','/usr/bin/gcc'])
_PROFILER='/opt/dtk/rocprofiler/lib/rocprofiler/librocprofiler_tool.so'
_LOG=pathlib.Path(os.environ['OUTPUT_PATH']).parent/'compiler-environments'
def _record(data):
 data.update({'pid':os.getpid(),'monotonic_ns':time.monotonic_ns()})
 try:
  _LOG.mkdir(parents=True,exist_ok=True)
  with (_LOG/(str(os.getpid())+'.jsonl')).open('a') as f:f.write(json.dumps(data)+'\n')
 except OSError:
  try:os.write(2,b'GLM53_COMPILER_ENV_LOG_ERROR\n')
  except OSError:pass
def _clean_child_environment(environment):
 source=os.environ if environment is None else environment
 key='LD_PRELOAD' if 'LD_PRELOAD' in source else b'LD_PRELOAD'
 value=source.get(key)
 if not isinstance(value,(str,bytes)):return None
 target=_PROFILER.encode() if isinstance(value,bytes) else _PROFILER
 pattern=rb'[^:\s]+' if isinstance(value,bytes) else r'[^:\s]+'
 tokens=re.findall(pattern,value)
 if target not in tokens:return None
 empty=b'' if isinstance(value,bytes) else ''
 remaining=[x for x in tokens if x!=target]
 child=dict(source)
 if remaining:child[key]=re.sub(pattern,lambda m:empty if m.group()==target else m.group(),value)
 else:child.pop(key,None)
 return child
class CompilerEnvironmentPopen(_ORIGINAL):
 @functools.wraps(_ORIGINAL.__init__)
 def __init__(self,*args,**kwargs):
  call=_SIGNATURE.bind(self,*args,**kwargs)
  command=call.arguments.get('args');executable=call.arguments.get('executable')
  if executable is None:
   executable=command[0] if isinstance(command,(list,tuple)) and command else command
  matched=False
  if not call.arguments.get('shell',False) and isinstance(executable,(str,bytes,os.PathLike)) and os.fsdecode(executable) in _COMPILERS:
   cleaned=_clean_child_environment(call.arguments.get('env'))
   if cleaned is not None:call.arguments['env']=cleaned;matched=True
  _ORIGINAL.__init__(*call.args,**call.kwargs)
  if matched:_record({'event':'compiler_spawned_with_profiler_removed','child_pid':self.pid,'executable':os.fsdecode(executable)})
subprocess.Popen=CompilerEnvironmentPopen
_record({'event':'python_hook_loaded','argv0':sys.argv[0],'profiler_mapped':_PROFILER in pathlib.Path('/proc/self/maps').read_text()})
