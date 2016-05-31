def normalize(values, v_min=0.0):
    for i in xrange(1, len(values)):
        values[i] = float(values[i])
    v_max = max(values[1:])
    new_values = []
    for i in values[1:]:
        if i >= v_max:
            new_values.append(i-v_min*9)
        else:
            new_values.append(i+v_min)
    return values[:1] + new_values


def evaluate(filename, ans_file="pa_test_new_label.csv"):
    import math
    fname2digit = {}
    logloss = []
    for line in open(ans_file):
        values = line.strip().split(",")
        fname2digit[values[0]] = int(values[1])
    for line in open(filename):
        values = normalize(line.strip().split(","), 0.0)
        try:
            loss = float(values[1+fname2digit[values[0]]])
            if loss == 0.0:
                logloss.append(-22)
            else:
                logloss.append(math.log(loss))
        except:
            pass
        else:
            continue
    return sum(logloss)/len(logloss)

if __name__ == "__main__":
    import sys
    print evaluate(sys.argv[1])


