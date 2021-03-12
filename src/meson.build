project('mesonTest', 'cuda', version : '10.0.0')

# cmake = import('cmake')

incdir = include_directories('../include/')

zed_dep = dependency('ZED', method : 'auto')
# zed_dep = dependency('ZED', version : '>=3', method : 'auto', modules : ['ZED::ZED'])

cuda_comp = meson.get_compiler('cuda')
cuda = import('unstable-cuda')

# lcm = dependency('lcm')
cuda = dependency('cuda', version : '>=10', modules : ['cublas'])
all_deps = [cuda, zed_dep]

exe = executable('obs-detector', 'obs-detector.cu', 'common.cu', 'euclidean-cluster.cu', 'find-clear.cu', 'GLViewer.cu', 'pass-through.cu', 
                  'pcl.cu', 'plane-ransac.cu', dependencies : all_deps, include_directories : incdir)

test('basicCudaTest', exe)
