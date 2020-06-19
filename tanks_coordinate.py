#/usr/bin/env python

def all_coordinate():
    # x and y coordinate of both tanks of all IceTop stations.
    positions = {1: {'A': [-265.5299987792969, -497.8949890136719],
          'B': [-255.6999969482422, -496.07000732421875]},
         2: {'A': [-140.36000061035156, -477.76499938964844],
          'B': [-130.63500213623047, -476.5749969482422]},
         3: {'A': [-27.72000026702881, -464.49000549316406],
          'B': [-20.394999504089355, -458.9750061035156]},
         4: {'A': [105.65499877929688, -438.7050018310547],
          'B': [115.26499938964844, -436.864990234375]},
         5: {'A': [214.73999786376953, -432.4100036621094],
          'B': [219.90499877929688, -424.68499755859375]},
         6: {'A': [356.6000061035156, -398.1300048828125],
          'B': [366.2149963378906, -398.3800048828125]},
         7: {'A': [-344.0500030517578, -400.4750061035156],
          'B': [-334.3249969482422, -399.0050048828125]},
         8: {'A': [-231.2249984741211, -389.260009765625],
          'B': [-224.18500518798828, -382.94500732421875]},
         9: {'A': [-85.19499969482422, -359.7300109863281],
          'B': [-76.06499862670898, -362.44000244140625]},
         10: {'A': [25.09500026702881, -342.4700012207031],
          'B': [34.52499961853027, -339.7449951171875]},
         11: {'A': [135.54000091552734, -353.385009765625],
          'B': [134.22000122070312, -343.47999572753906]},
         12: {'A': [279.114990234375, -300.4600067138672],
          'B': [288.75999450683594, -301.50999450683594]},
         13: {'A': [386.1050109863281, -290.32000732421875],
          'B': [393.4499969482422, -283.77000427246094]},
         14: {'A': [-390.0349884033203, -335.68499755859375],
          'B': [-395.4250030517578, -343.8999938964844]},
         15: {'A': [-265.69500732421875, -305.0],
          'B': [-266.8699951171875, -314.8000030517578]},
         16: {'A': [-182.25, -268.1199951171875],
          'B': [-173.01499938964844, -263.8399963378906]},
         17: {'A': [-40.35499954223633, -243.04500579833984],
          'B': [-30.989999771118164, -246.06999969482422]},
         18: {'A': [66.38000106811523, -227.1699981689453],
          'B': [75.67999649047852, -223.73500061035156]},
         19: {'A': [210.22999572753906, -185.0999984741211],
          'B': [220.04000091552734, -186.24500274658203]},
         20: {'A': [311.4750061035156, -228.47000122070312],
          'B': [305.2749938964844, -220.85999298095703]},
         21: {'A': [441.7050018310547, -212.6750030517578],
          'B': [432.9649963378906, -207.02499389648438]},
         22: {'A': [-468.95001220703125, -238.43499755859375],
          'B': [-474.2099914550781, -246.7699966430664]},
         23: {'A': [-388.9750061035156, -194.84000396728516],
          'B': [-381.31500244140625, -188.66000366210938]},
         24: {'A': [-225.47000122070312, -175.76000213623047],
          'B': [-221.37000274658203, -184.4800033569336]},
         25: {'A': [-102.53499984741211, -155.80500030517578],
          'B': [-98.30500030517578, -164.68500518798828]},
         26: {'A': [13.460000038146973, -129.35499572753906],
          'B': [20.950000762939453, -135.9149932861328]},
         27: {'A': [125.58499908447266, -106.19500350952148],
          'B': [134.95999908447266, -108.70999908447266]},
         28: {'A': [228.75, -95.84000015258789],
          'B': [236.12999725341797, -89.84000015258789]},
         29: {'A': [363.385009765625, -115.68500137329102],
          'B': [354.7449951171875, -110.06000137329102]},
         30: {'A': [523.6900024414062, -67.34500122070312],
          'B': [518.1400146484375, -76.07500076293945]},
         31: {'A': [-547.9249877929688, -141.43499755859375],
          'B': [-553.510009765625, -149.5]},
         32: {'A': [-452.6199951171875, -88.37999725341797],
          'B': [-442.77500915527344, -88.6150016784668]},
         33: {'A': [-304.55999755859375, -78.4900016784668],
          'B': [-300.55999755859375, -87.37000274658203]},
         34: {'A': [-181.05500030517578, -58.75],
          'B': [-176.3699951171875, -68.21500015258789]},
         35: {'A': [-86.4000015258789, -30.985000610351562],
          'B': [-76.4900016784668, -29.475000381469727]},
         36: {'A': [37.435001373291016, -57.36000061035156],
          'B': [29.18000030517578, -52.80500030517578]},
         37: {'A': [150.77000427246094, -31.0649995803833],
          'B': [145.34500122070312, -22.655000686645508]},
         38: {'A': [316.0350036621094, -3.805000066757202],
          'B': [311.0749969482422, -12.414999961853027]},
         39: {'A': [434.9700012207031, 3.940000057220459],
          'B': [429.56500244140625, -4.5849997997283936]},
         40: {'A': [530.6050109863281, 20.969999313354492],
          'B': [522.0249938964844, 25.59000015258789]},
         41: {'A': [-506.7899932861328, -0.8450000286102295],
          'B': [-502.2250061035156, -9.755000114440918]},
         42: {'A': [-383.5399932861328, 18.315000534057617],
          'B': [-379.10499572753906, 9.704999923706055]},
         43: {'A': [-260.27000427246094, 38.61000061035156],
          'B': [-255.7800064086914, 29.114999771118164]},
         44: {'A': [-131.47000122070312, 38.26499938964844],
          'B': [-134.97000122070312, 29.4350004196167]},
         45: {'A': [-43.14999961853027, 40.46500015258789],
          'B': [-51.1299991607666, 46.255001068115234]},
         46: {'A': [71.3499984741211, 66.82499694824219],
          'B': [67.07500076293945, 75.39500045776367]},
         47: {'A': [172.68000030517578, 114.55500030517578],
          'B': [170.2750015258789, 124.2249984741211]},
         48: {'A': [365.4499969482422, 151.48500061035156],
          'B': [357.0449981689453, 156.81999969482422]},
         49: {'A': [495.2200012207031, 136.94000244140625],
          'B': [496.76499938964844, 127.48500061035156]},
         50: {'A': [604.1549987792969, 147.50999450683594],
          'B': [596.7449951171875, 140.6050033569336]},
         51: {'A': [-456.8450012207031, 101.56500244140625],
          'B': [-458.26499938964844, 91.2249984741211]},
         52: {'A': [-333.239990234375, 120.63999938964844],
          'B': [-334.51499938964844, 111.43000030517578]},
         53: {'A': [-210.09500122070312, 138.9499969482422],
          'B': [-212.11499786376953, 129.81499481201172]},
         54: {'A': [-121.7599983215332, 136.53500366210938],
          'B': [-129.40499877929688, 142.47000122070312]},
         55: {'A': [21.235000610351562, 155.66500091552734],
          'B': [11.3100004196167, 153.7249984741211]},
         56: {'A': [133.06500244140625, 174.70999908447266],
          'B': [123.9900016784668, 177.21500396728516]},
         57: {'A': [248.7449951171875, 194.55999755859375],
          'B': [241.1999969482422, 200.65499877929688]},
         58: {'A': [400.48500061035156, 220.52499389648438],
          'B': [392.9199981689453, 215.27999877929688]},
         59: {'A': [525.2449951171875, 241.56499481201172],
          'B': [517.6300048828125, 235.4199981689453]},
         60: {'A': [-416.14500427246094, 231.43499755859375],
          'B': [-412.50498962402344, 222.2899932861328]},
         61: {'A': [-288.72999572753906, 232.7300033569336],
          'B': [-292.4250030517578, 224.0]},
         62: {'A': [-201.68999481201172, 234.9800033569336],
          'B': [-209.19499969482422, 241.20499420166016]},
         63: {'A': [-53.35500144958496, 255.73500061035156],
          'B': [-61.53499984741211, 252.5500030517578]},
         64: {'A': [64.76499938964844, 272.45001220703125],
          'B': [55.30999946594238, 271.0449981689453]},
         65: {'A': [153.6050033569336, 302.9650115966797],
          'B': [152.34500122070312, 312.2550048828125]},
         66: {'A': [319.9449920654297, 316.94500732421875],
          'B': [312.5800018310547, 312.5350036621094]},
         67: {'A': [440.3050079345703, 333.4150085449219],
          'B': [431.0349884033203, 330.5749969482422]},
         68: {'A': [-403.48500061035156, 312.2050018310547],
          'B': [-410.86500549316406, 317.40501403808594]},
         69: {'A': [-279.64500427246094, 331.15501403808594],
          'B': [-287.27500915527344, 336.4550018310547]},
         70: {'A': [-161.46499633789062, 354.4250030517578],
          'B': [-167.69499969482422, 361.5950012207031]},
         71: {'A': [-12.585000038146973, 370.239990234375],
          'B': [-21.90999984741211, 367.55999755859375]},
         72: {'A': [123.5150032043457, 400.9149932861328],
          'B': [116.6500015258789, 393.50999450683594]},
         73: {'A': [207.94499969482422, 413.01499938964844],
          'B': [202.02999877929688, 421.2949981689453]},
         74: {'A': [369.3199920654297, 431.0],
          'B': [362.0900115966797, 425.4100036621094]},
         75: {'A': [-352.60499572753906, 421.4250030517578],
          'B': [-361.2449951171875, 424.1700134277344]},
         76: {'A': [-241.86000061035156, 453.32000732421875],
          'B': [-247.61000061035156, 461.43499755859375]},
         77: {'A': [-87.70499801635742, 468.72499084472656],
          'B': [-97.10000228881836, 465.1949920654297]},
         78: {'A': [2.10999995470047, 494.6199951171875],
          'B': [-2.0199999809265137, 503.5950012207031]},
         79: {'A': [18.09999942779541, -94.63000106811523],
          'B': [10.880000114440918, -87.69499969482422]},
         80: {'A': [76.42499923706055, -42.47999954223633],
          'B': [85.64500045776367, -46.0]},
         81: {'A': [77.40999984741211, 37.6150016784668],
          'B': [87.19000244140625, 39.84000015258789]}}

    return positions

    