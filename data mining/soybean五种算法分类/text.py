import matplotlib.pyplot as plt
import numpy as np
import joblib
from soybean_classification import accuracy_score, all_Algorithms
from datapreprocessing import load_data_set_test
from matplotlib.font_manager import FontProperties


def showbar(all_algorithms_name, all_algorithms_score, all_algorithms_score_avg):
    font = FontProperties('simhei', size=14)
    x = np.arange(5)
    all_algorithms_score_avg = [-l for l in all_algorithms_score_avg]
    plt.bar(x, all_algorithms_score, width=0.3, label="vaildate_score")
    plt.bar(x, all_algorithms_score_avg, width=0.3, label="avg_score")
    plt.legend(bbox_to_anchor=(1.01, 1),
               ncol=1,
               mode="None",
               borderaxespad=0,
               shadow=False,
               fancybox=True)
    plt.xticks(np.arange(5), all_algorithms_name, rotation=90)  # rotation控制倾斜角度
    plt.xlabel(u'algorithms', fontproperties=font)
    plt.ylabel(u'score', fontproperties=font)
    plt.title(u'算法结果', fontproperties=font)
    plt.savefig('result.png')
    plt.show()


all_algorithms_name, all_algorithms_score, all_algorithms_score_avg, df_name, df_col_mean = all_Algorithms()
showbar(all_algorithms_name, all_algorithms_score, all_algorithms_score_avg)