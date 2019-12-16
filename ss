if cond == -1 and cond_way == -1:               # kiri - kiri
            wx[0] = path.x[1] - adder_lane - adder_way
            wy[0] = path.y[1]
            print("1")
        elif cond == 1 and cond_way == 1:               # kanan - kanan
            wx[0] = path.x[1] + adder_lane + adder_way
            wy[0] = path.y[1]
            print("2")
        elif cond == -1 and cond_way == 1:              # lane --> kiri , way --> kanan
            wx[0] = path.x[1] - adder_lane  
            wy[0] = path.y[1]
            print("3")
        elif cond == 1 and cond_way == -1:              # lane --> kanan , way --> kiri
            wx[0] = path.x[1] + adder_lane
            wy[0] = path.y[1]
            print("4")
        elif cond == 0 and cond_way == 0:               # lurus - lurus
            wx[0] = path.x[1]
            wy[0] = path.y[1]
            print("5")
        elif cond == 0 and cond_way == -1:              # lurus - kiri
            wx[0] = path.x[1] - adder_way
            wy[0] = path.y[1]
            print("6")
        elif cond == 0 and cond_way == 1:               # lurus - kanan
            wx[0] = path.x[1] + adder_way
            wy[0] = path.y[1]
            print("7")
        elif cond == -1 and cond_way == 0:              # kiri - lurus
            wx[0] = path.x[1] - adder_lane
            wy[0] = path.y[1]
            print("8")
        elif cond == 1 and cond_way == 0:               # kanan - lurus
            wx[0] = path.x[1] + adder_lane
            wy[0] = path.y[1]
            print("9")
