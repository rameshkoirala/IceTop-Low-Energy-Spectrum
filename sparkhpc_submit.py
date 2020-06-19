#!/usr/bin/env python

from sparkhpc import sparkjob
from optparse import OptionParser
import time

parser = OptionParser(usage="%prog [OPTIONS]", description=__doc__)
parser.add_option("--ncores", "-n", default=40, type=int, help="Number of cores.")
opts, args = parser.parse_args()

# https://github.com/rokroskar/sparkhpc/blob/master/sparkhpc/sparkjob.py
# https://github.com/rokroskar/sparkhpc/blob/master/example.ipynb
# ncores=1, cores_per_executor=1 , time=12:12
# ncores=8, cores_per_executor=8 , time=2:36
# ncores=16, cores_per_executor=8, time=1:47
sj = sparkjob.sparkjob(ncores=opts.ncores,
					cores_per_executor=12, #does not work without this # 5
					walltime='12:00',
					memory_per_core=1500, #this needs to be as high as possible. #3072
					spark_home='/home/rkoirala/spark',
					scheduler='slurm'
					)
sj.submit()
time.sleep(10)
print 'Master: ', sj.master_url()

# Asterix (excluding interactive partition):
#   nodes: 35
#   cores: 328
#   memory: 760 GB

#At terminal: $SPARK_HOME/bin/spark-submit --master spark://asterix-node04:7077 \
#                                          --executor-memory 9G \
#                                          --total-executor-cores 50 \
#                                               spark_cv.py