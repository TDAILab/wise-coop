import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score
import copy


class Cooperante(object):
    def __init__(self, penalty_matrix, forbidden=[]):
        assert penalty_matrix.shape[0] == penalty_matrix.shape[1]
        assert type(forbidden) == list #contain index of forbidden class
        self.penalty_matrix = penalty_matrix
        self.num_class = self.penalty_matrix.shape[1]
        self.penalty_matrix = penalty_matrix
        assert type(forbidden) == list
        self.forbidden = forbidden
        all_class = set(range(0,self.num_class))     
        unforbidden = all_class - set(forbidden)
        self.unforbidden = list(unforbidden) #[0,1,3]みたいな特定の列を除いたもの
        # d = dict(enumerate(unforbidden)) #{0:0, 1:1, 2:3}　みたいな辞書
        # d1 = {v:k for k,v in d.items()} #{0:0, 1:1, 3:2} にkeyとvalueの順序を入れ替えた
        # self.adjuster = d1
        self.check_rate_df = pd.DataFrame([90, 95, 99], columns = ["score over X %"])

        """
        penalty matrix
          a b c d
        a 0 1 1 1
        b 9 0 1 1
        c 9 1 0 1
        d 1 1 1 0

        probability_array
        a b c d
        0.1 0.2 0.3 0.4
        0.7 0.1 0.1 0.1
        .
        .
        .

        """

    def fit(self, probability_array)->np.array:
        assert probability_array.shape[1] == self.penalty_matrix.shape[1]

        #caluculate mu_penalty for each label
        mu_min_array = np.array([])
        mu_min_index_array = np.array([])

        mu_for_each_label = np.dot(probability_array, self.penalty_matrix).astype(float) #内積をとる #あとでnanを入れたいのでfloatにしておく
        mu_for_each_label[:, self.forbidden] = np.nan  #指定したラベルの列にnanを挿入
        mu_min = np.nanmin(mu_for_each_label, axis=1) #最小のペナルティー期待値
        mu_min_index = np.nanargmin(mu_for_each_label, axis=1) #これが選ぶラベル。ただしforbiddenで指定された列を抜いていることに注意
        mu_min_array = np.append(mu_min_array, mu_min)
        mu_min_index_array = np.append(mu_min_index_array, mu_min_index)        
        prediction = mu_min_index_array.astype(int) #予測ラベルがfloatになってるのでintに戻す
        df = pd.DataFrame(prediction, columns=["pred"])  
        df["mu_min"] = mu_min_array
        self.df = df

        return mu_min_array, prediction

    def plot_eval(self, label_array, metrics, show_oracle=False, samplong_rate = 5):
        assert type(metrics) == list
        self._sampling_rate = samplong_rate
        self.num_sample = int(100 / samplong_rate + 1)
        self.df["ans"] = label_array
        self.emerging_label = sorted(list(set(label_array))) #全てのラベルが必ずしも正解ラベルにあるわけではない
        sorted_df = self.df.sort_values("mu_min", ascending=False).reset_index()
        self.scores = self._calc_3scores(sorted_df, metrics)

        
        num_plot = len(metrics)
        plt.figure(figsize=(6,5*num_plot))
        for i,x in enumerate(metrics):
            plt.subplot(num_plot,1,i+1)
            for j,y in enumerate(self.emerging_label): #各クラスを陽性ラベルとしたときのラインをかく
                if x == "accuracy_score":
                    plt.plot(np.linspace(0,100,self.num_sample), self.scores[x], label = "label" + str(y)) 
                    if show_oracle:
                        plt.plot(np.linspace(0,100,self.num_sample), self._calc_score_oracle(self.df, x, y), label = "oracle_" + "label" + str(y)) 
                    self.check_rate_df[f"{x}(label{y})"] = self.check_rates(self.scores[x])
                else:
                    plt.plot(np.linspace(0,100,self.num_sample), self.scores[x][:,j], label = "label" + str(y)) 
                    if show_oracle:
                        plt.plot(np.linspace(0,100,self.num_sample), self._calc_score_oracle(self.df, x, y)[:,j], label = "oracle_" + "label" + str(y)) 
                    self.check_rate_df[f"{x}(label{y})"] = self.check_rates(self.scores[x][:,j])
                
            plt.legend()
            plt.xlabel('Human Check Percent')
            plt.xlim(0,100)
            plt.grid()

    def _calc_score(self, n, df, score_type = "f1_score")->list: #dataframeのn番目までhuman checkしてスコアを返す  
        n = int(n)
        dataframe = df.copy()
        
        dataframe.loc[:n, "pred"] =dataframe.loc[:n, "ans"]
        if score_type == "accuracy_score":
            score = accuracy_score(dataframe["ans"],dataframe["pred"])
        elif score_type == "f1_score":
            score = f1_score(dataframe["ans"],dataframe["pred"], average = None, labels=self.emerging_label)
        elif score_type == "precision_score":
            score = precision_score(dataframe["ans"],dataframe["pred"],  average = None, labels=self.emerging_label)
        elif score_type == "recall_score":
            score = recall_score(dataframe["ans"],dataframe["pred"],  average = None, labels=self.emerging_label)
        return score.tolist() #len(score) = len(self.num_class), type(score) = ndarray [0.1 0.1 0.4]みたいなのが入ってる
            
    def _calc_3scores(self, sorted_df, metrics)->dict:
        scores_dict = {}
        for x_score in metrics:
            scores_dict[x_score] = np.array([self._calc_score(n, sorted_df, score_type=x_score) for n in np.linspace(0,sorted_df.shape[0], self.num_sample)]) #shape は(データ数, num_class)
        return scores_dict

    def _calc_score_oracle(self, df, score_type = "f1_score", label = 1)->np.array:
        #正解ラベル、予測ラベルのついたdf
        #sort
        pred_negative = df[df["pred"] == label]
        ans_negative = df[df["ans"] == label]

        if score_type == "precision_score":
            wrong_idx = pred_negative[pred_negative["ans"] != pred_negative["pred"]].index
            else_idx = df.drop(index = wrong_idx).index

        elif score_type == "recall_score":
            wrong_idx = ans_negative[ans_negative["ans"] != ans_negative["pred"]].index
            else_idx = df.drop(index = wrong_idx).index

        elif score_type  == "f1_score":
            wrong_idx_precision = pred_negative[pred_negative["ans"] != pred_negative["pred"]].index
            wrong_idx_recall = ans_negative[ans_negative["ans"] != ans_negative["pred"]].index
            wrong_idx = wrong_idx_precision.append(wrong_idx_recall)
            else_idx = df.drop(index = wrong_idx).index

        elif score_type == "accuracy_score":
            wrong_idx_precision = pred_negative[pred_negative["ans"] != pred_negative["pred"]].index
            wrong_idx_recall = ans_negative[ans_negative["ans"] != ans_negative["pred"]].index
            wrong_idx = wrong_idx_precision.append(wrong_idx_recall)
            else_idx = df.drop(index = wrong_idx).index

        oracle_idx = wrong_idx.append(else_idx)
        df_oraclesort = df.loc[oracle_idx, :].reset_index()  #間違ってるものを上に持ってきた
        oracle_score = [self._calc_score(n, df_oraclesort, score_type) for n in np.linspace(0,df_oraclesort.shape[0], self.num_sample)]

        return np.array(oracle_score) #[[...],[...],...,[...]] #各行のk番目にラベルkの時のスコアが入っている

    # def _calc_3scores_oracle(self, df, metrics, label):
    #     scores_dict = {}
    #     for x_score in metrics:
    #         scores_dict[x_score] = np.array([self._calc_score_oracle(n, df, score_type=x_score, label = label) for n in tqdm(np.linspace(0,df.shape[0], 21))]) #shape は(データ数, num_class)
    #     return scores_dict

    def check_rate(self, threshold, score_array):
        score_list = list(score_array)
        score = score_array[score_array >= threshold][0] #初めて閾値を超えた時のスコア
        check_rate = score_list.index(score)
        return check_rate * self._sampling_rate

    def check_rates(self, score_list):
        thresholds_list = [0.90, 0.95, 0.99]
        check_rates = []
        for threshold in thresholds_list:
            check_rates.append(self.check_rate(threshold, score_list))
        return check_rates
        