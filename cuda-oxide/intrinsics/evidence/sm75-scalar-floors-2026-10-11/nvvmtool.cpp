#include <nvvm.h>
#include <nvJitLink.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string slurp(const char* p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }
static void spit(const std::string& p, const std::string& d) { std::ofstream f(p, std::ios::binary); f.write(d.data(), d.size()); }

static std::string nvvm_compile(const std::string& ir, const std::vector<std::string>& opts) {
  nvvmProgram prog; nvvmCreateProgram(&prog);
  if (nvvmAddModuleToProgram(prog, ir.data(), ir.size(), "m") != NVVM_SUCCESS) { fprintf(stderr, "add module failed\n"); exit(2); }
  std::vector<const char*> o; for (auto& s : opts) o.push_back(s.c_str());
  nvvmResult r = nvvmCompileProgram(prog, (int)o.size(), o.data());
  size_t n; nvvmGetProgramLogSize(prog, &n); std::string log(n, 0); nvvmGetProgramLog(prog, &log[0]);
  if (r != NVVM_SUCCESS) { fprintf(stderr, "nvvm compile failed: %s\n", log.c_str()); exit(3); }
  nvvmGetCompiledResultSize(prog, &n); std::string out(n, 0); nvvmGetCompiledResult(prog, &out[0]);
  nvvmDestroyProgram(&prog);
  return out;
}

static std::string link(const std::string& ltoir, bool ptx) {
  std::vector<std::string> opts = {"-arch=sm_75", "-lto"};
  if (ptx) opts.push_back("-ptx");
  std::vector<const char*> o; for (auto& s : opts) o.push_back(s.c_str());
  nvJitLinkHandle h;
  if (nvJitLinkCreate(&h, (uint32_t)o.size(), o.data()) != NVJITLINK_SUCCESS) { fprintf(stderr, "create failed\n"); exit(4); }
  nvJitLinkAddData(h, NVJITLINK_INPUT_LTOIR, ltoir.data(), ltoir.size(), "ltoir");
  if (nvJitLinkComplete(h) != NVJITLINK_SUCCESS) {
    size_t n; nvJitLinkGetErrorLogSize(h, &n); std::string l(n, 0); nvJitLinkGetErrorLog(h, &l[0]);
    fprintf(stderr, "link failed: %s\n", l.c_str()); exit(5);
  }
  size_t n = 0; std::string out;
  if (ptx) { nvJitLinkGetLinkedPtxSize(h, &n); out.assign(n, 0); nvJitLinkGetLinkedPtx(h, &out[0]); }
  else { nvJitLinkGetLinkedCubinSize(h, &n); out.assign(n, 0); nvJitLinkGetLinkedCubin(h, &out[0]); }
  nvJitLinkDestroy(&h);
  return out;
}

int main(int argc, char** argv) {
  if (argc != 3) { fprintf(stderr, "usage: nvvmtool in.ll outprefix\n"); return 1; }
  std::string ir = slurp(argv[1]); std::string pre = argv[2];
  std::string ptx = nvvm_compile(ir, {"-arch=compute_75"});
  while (!ptx.empty() && ptx.back() == 0) ptx.pop_back();
  spit(pre + ".ptx", ptx);
  std::string lto = nvvm_compile(ir, {"-arch=compute_75", "-gen-lto"});
  spit(pre + ".ltoir", lto);
  spit(pre + ".linked.cubin", link(lto, false));
  std::string lp = link(lto, true);
  while (!lp.empty() && lp.back() == 0) lp.pop_back();
  spit(pre + ".linked.ptx", lp);
  return 0;
}
