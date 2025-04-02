import csv
# 设置支持度计数
min_sup = 100
# 设置置信度
min_conf = 0.6
# 设置最大K项集，结果可以输出至前K项集
K = 3


# 载入数据
def load_data():
    with open(r"C:\Users\ljz.2002\Desktop\py\pythonProject\数据挖掘\Groceries_dataset.csv", 'r') as fl:
        reader = csv.reader(fl)
        header_row = next(reader)
        data = {}  # 创建空字典
        for row in reader:  # 逐行读取表格数据
            if int(row[0]) not in data.keys():  # 如果人（即编号）不在字典中，创建键值对
                data[int(row[0])] = [row[2]]
            else:
                data[int(row[0])].append(row[2])  # 如果在，将物品添加进value列表
        d = []
        for key in data:
            d.append(data[key])
        return d


def apriori():
    d = load_data()  # 载入数据
    C1 = find_frequent_1itemsets(d)  # 计算支持数
    item_count = count_itemset1(d, C1)
    L1 = generate_L1(item_count)   # 剪枝，去掉支持数小于最小支持度数min_sup的项
    Lk_copy = L1.copy()    # 连接
    L = []
    L.append(Lk_copy)
    for i in range(2, K + 1):    # 7.重复连接、剪枝、直到得到K频繁项集
        Ci = create_Ck(Lk_copy, i)
        Li = generate_Lk_by_Ck(Ci, d)
        Lk_copy = Li.copy()
        L.append(Lk_copy)
    print('频繁项集\t支持度计数')  # 输出频繁项集及其支持度数
    support_data = {}
    for item in L:
        for i in item:
            print(list(i), '\t', item[i])
            support_data[i] = item[i]
    strong_rules_list = generate_strong_rules(L, support_data, d)    # 生成强关联规则
    strong_rules_list.sort(key=lambda result: result[2], reverse=True)
    print("\nStrong association rule\nX\t\t\tY\t\tconf")
    for item in strong_rules_list:
        print(list(item[0]), "\t", list(item[1]), "\t", item[2])


# 生成频繁1项集
def find_frequent_1itemsets(d):
    C1 = set()  # 创建集合C1,集合类型的数据具有唯一性
    for each in d:  # 在储存了字典所有的值的列表d中
        for item in each:  # 对其中每个列表（即每个value值）
            item_set = frozenset([item])  # 将数据冻结在集合中，不可增删
            C1.add(item_set)  # 添加到C1中
    return C1


# 计算给定数据每项及其支持数
def count_itemset1(d, C1):
    item_count = {}
    for data in d:
        for item in C1:
            if item.issubset(data):
                if item in item_count:
                    item_count[item] += 1
                else:
                    item_count[item] = 1
    return item_count


# 生成剪枝后的L1
def generate_L1(item_count):
    L1 = {}
    for i in item_count:
        if item_count[i] >= min_sup:
            L1[i] = item_count[i]
    return L1


# 判断是否剪枝
def is_apriori(Ck_item, Lk_copy):
    for item in Ck_item:
        sub_Ck = Ck_item - frozenset([item])
        if sub_Ck not in Lk_copy:
            return False
    return True


# 生成K项商品集，连接
def create_Ck(Lk_copy, k):
    Ck = set()
    len_Lk_copy = len(Lk_copy)
    list_Lk_copy = list(Lk_copy)
    for i in range(len_Lk_copy):
        for j in range(1, len_Lk_copy):
            l1 = list(list_Lk_copy[i])
            l2 = list(list_Lk_copy[j])
            l1.sort()
            l2.sort()
            if l1[0:k-2] == l2[0:k-2]:
                Ck_item = list_Lk_copy[i] | list_Lk_copy[j]
                # 扫描前一个项集，剪枝
                if is_apriori(Ck_item, Lk_copy):
                    Ck.add(Ck_item)
    return Ck


# 生成剪枝后的Lk
def generate_Lk_by_Ck(Ck, data_set):
    item_count = {}
    for data in data_set:
        for item in Ck:
            if item.issubset(data):
                if item in item_count:
                    item_count[item] += 1
                else:
                    item_count[item] = 1
    Lk2 = {}
    for i in item_count:
        if item_count[i] >= min_sup:
            Lk2[i] = item_count[i]
    return Lk2


# 生成强关联规则
def generate_strong_rules(L, support_data, d):
    strong_rule_list = []
    sub_set_list = []
    # print(L)
    for i in range(0, len(L)):
        for freq_set in L[i]:
            for sub_set in sub_set_list:
                if sub_set.issubset(freq_set):
                    # 计算包含X的交易数
                    sub_set_num = 0
                    for item in d:
                        if (freq_set - sub_set).issubset(item):
                            sub_set_num += 1
                    conf = support_data[freq_set] / sub_set_num
                    strong_rule = (freq_set - sub_set, sub_set, conf)
                    if conf >= min_conf and strong_rule not in strong_rule_list:
                        strong_rule_list.append(strong_rule)
            sub_set_list.append(freq_set)
    return strong_rule_list


if __name__ == '__main__':
    apriori()
