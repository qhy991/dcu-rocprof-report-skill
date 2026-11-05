import os,sys,pathlib,subprocess,time,json,hashlib,ctypes,statistics
import sitecustomize as hook

if not __debug__:
    raise SystemExit("Run this acceptance test without Python optimization.")
w=pathlib.Path(sys.argv[1]);cache=pathlib.Path(sys.argv[2]);tool=hook._PROFILER;gcc='/usr/bin/gcc';clang='/opt/dtk/aillvm/bin/clang-22'
w.mkdir(exist_ok=False)
if not cache.is_dir():raise SystemExit('The native compiler cache is not a directory.')
assert tool in pathlib.Path('/proc/self/maps').read_text()
source=w/'value.c';source.write_text('int value(int x) { return x * 3 + 1; }\n')
extra=w/'extra.so';subprocess.run([gcc,'-shared','-fPIC',str(source),'-o',str(extra)],check=True,timeout=60)
env=os.environ.copy();env['LD_PRELOAD']=str(extra)+' : '+tool
parent=dict(os.environ);given=dict(env);evidence=[]
def maps_probe(kind):
 old=os.environ.get('LD_PRELOAD')
 if kind=='none':os.environ['LD_PRELOAD']=env['LD_PRELOAD']
 before=dict(os.environ);p=None
 try:
  if kind=='none':p=subprocess.Popen([gcc,'-E','-x','c','-'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  elif kind=='positional':p=subprocess.Popen([gcc,'-E','-x','c','-'],-1,None,subprocess.PIPE,subprocess.PIPE,subprocess.PIPE,None,True,False,None,env)
  else:p=subprocess.Popen([gcc,'-E','-x','c','-'],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  time.sleep(.2);mapped=pathlib.Path(f'/proc/{p.pid}/maps').read_text();assert tool not in mapped;assert str(extra) in mapped;assert dict(os.environ)==before
  out,err=p.communicate(b'int sample;\\n',timeout=10);assert p.returncode==0 and b'sample' in out
  evidence.append({'case':kind,'profiler_mapped':False,'foreign_preload_mapped':True,'returncode':p.returncode})
 finally:
  if p is not None and p.poll() is None:p.kill();p.communicate()
  if kind=='none':
   if old is None:os.environ.pop('LD_PRELOAD',None)
   else:os.environ['LD_PRELOAD']=old
for kind in ['keyword','positional','none']:maps_probe(kind)
assert os.environ==parent and env==given
raw=hook._ORIGINAL([gcc,'-E','-x','c','-'],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
try:
 time.sleep(.2);assert tool in pathlib.Path(f'/proc/{raw.pid}/maps').read_text();raw.communicate(b'int original;\n',timeout=10);assert raw.returncode==0
finally:
 if raw.poll() is None:raw.kill();raw.communicate()
evidence.append({'case':'original compiler control','profiler_mapped':True})
python_probe='import pathlib;print('+repr(tool)+' in pathlib.Path("/proc/self/maps").read_text())'
assert subprocess.check_output([sys.executable,'-c',python_probe],env=env,text=True).strip()=='True'
assert subprocess.check_output([clang,'-c',python_probe],executable=sys.executable,env=env,text=True).strip()=='True'
assert 'clang version' in subprocess.check_output([sys.executable,'--version'],executable=clang,env=env,text=True).lower()
with subprocess.Popen(['/bin/echo','popen-api'],env=env,stdout=subprocess.PIPE) as p:assert p.communicate()[0].strip()==b'popen-api' and p.wait()==0
try:subprocess.run([sys.executable,'-c','raise SystemExit(7)'],env=env,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
except subprocess.CalledProcessError as exc:assert exc.returncode==7
else:raise AssertionError('nonzero exit lost')
try:subprocess.run([sys.executable,'-c','import time;time.sleep(5)'],env=env,timeout=2,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
except subprocess.TimeoutExpired:pass
else:raise AssertionError('timeout lost')
saved_log=hook._LOG;hook._LOG=pathlib.Path('/proc/invalid-compiler-log-directory')
try:
 with subprocess.Popen([clang,'--version'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as p:p.communicate(timeout=10);assert p.returncode==0
finally:hook._LOG=saved_log
artifacts=[]
def compare_compile(name,command,output):
 hashes=[]
 for filtered in [False,True]:
  ctor=subprocess.Popen if filtered else hook._ORIGINAL
  with ctor(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as p:
   try:stdout,stderr=p.communicate(timeout=60)
   except subprocess.TimeoutExpired:p.kill();p.communicate();raise
   assert p.returncode==0,(command,stderr.decode())
  hashes.append(hashlib.sha256(output.read_bytes()).hexdigest())
 assert hashes[0]==hashes[1],(name,hashes)
 artifacts.append({'name':name,'command':command,'unfiltered_sha256':hashes[0],'filtered_sha256':hashes[1]})
lib=w/'compiled.so';compare_compile('C shared library',[gcc,'-O2','-shared','-fPIC',str(source),'-o',str(lib)],lib)
assert ctypes.CDLL(str(lib)).value(13)==40
asm=min(cache.rglob('*.amdgcn'),key=lambda p:p.stat().st_size);input_asm=w/'gpu-input.s';input_asm.write_bytes(asm.read_bytes());output=w/'gpu-output.hsaco'
flags=['-target','amdgcn-amd-amdhsa','-mcpu=gfx938:xnack-','-mllvm=-check-valu-data-forward-hazards=0','-mllvm=-disable-cluster-lds-memops=true','-mllvm=-hcu-pre-emit-load-store-opt=false','-mllvm=-support-768-vgprs=true','-mllvm=-amdgpu-disable-backoff-barrier=false','-mllvm=-fp-fuse=fast','-O3']
compare_compile('AMD GPU object CPU assembler',[clang,*flags,str(input_asm),'-x','assembler','-o',str(output)],output)
samples=[]
for i in range(5):
 for filtered in [False,True]:
  ctor=subprocess.Popen if filtered else hook._ORIGINAL;start=time.monotonic()
  with ctor([clang,'--version'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as p:
   try:p.communicate(timeout=30)
   except subprocess.TimeoutExpired:p.kill();p.communicate();raise
   assert p.returncode==0
  samples.append({'pair':i,'filtered':filtered,'seconds':time.monotonic()-start,'returncode':p.returncode})
assert os.environ==parent and env==given
result={'accepted':True,'scope':'CPU compiler environment boundary only; no GPU queue or kernel submitted','process_mapping_evidence':evidence,'recorder_io_failure_preserves_spawn':True,'ordinary_python_keeps_profiler':True,'executable_override_checked':True,'parent_and_supplied_env_unchanged':True,'Popen_pipe_context_exit_timeout_checked':True,'artifacts':artifacts,'clang_version_samples':samples,'medians':{str(mode):statistics.median(x['seconds'] for x in samples if x['filtered']==mode) for mode in [False,True]}}
(w/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
