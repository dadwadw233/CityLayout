from CityDM import CityDM
import pstats, cProfile

path = "/home/admin/workspace/yuyuanhong/code/CityLayout/config/new/uniDM_train.yaml"

citydm = CityDM(path)
# analyse the time cost of the function
cProfile.runctx('citydm.train()', globals(), locals(), './Profile.prof')

# sort the profile by time cost
s = pstats.Stats('./Profile.prof')
s.strip_dirs().sort_stats('time').print_stats()
