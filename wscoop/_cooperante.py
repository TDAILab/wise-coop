import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score
import copy


class Cooperante(object):
    def __init__(self, penalty_matrix, forbidden=[]):
        """
        Parameters
        ----------
        penalty_matrix : ndarray of shape (n_classes, n_classes)
            Penalty_matrix[i, j] means that when class i is predicted as class j, the penalty will be added. You can avoid some mis-predicting
            by setting the penalty high. Penalty_matrix[i, i] should be 0.
        
        forbidden : list, default=[]
            When you want to exclude some class from prediction, the class number should be included here as int. 
        Notes
        -----
        
        """
        
        assert penalty_matrix.shape[0] == penalty_matrix.shape[1]
        assert type(forbidden) == list #contain index of forbidden class
        self.penalty_matrix = penalty_matrix
        self.n_classes = self.penalty_matrix.shape[1]
        self.penalty_matrix = penalty_matrix
        assert type(forbidden) == list
        self.forbidden = forbidden
        all_class = set(range(0,self.n_classes))     
        unforbidden = all_class - set(forbidden)
        self.unforbidden = list(unforbidden) #[0,1,3]みたいな特定の列を除いたもの
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
        """
        Parameters
        ----------
        probability_array : ndarray of shape (n_samples, n_classes)
            The output from some classifier should be here.
       
        Notes
        -----
        When you use binary classifucation result as input here, the shape may be (n_samples, ). 

        Returns
        _______
        mu_min_array : 1darray of shape (n_samples, )
            The minimum penalties of the input samples. Higher penalty means the sample has to be
            checked by human.
        
        prediction : 1darray of shape (n_samples, )
            Class number is predicted for each sample so that the class has the minimum penalty.

        """
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

    def plot_eval(self, label_array, metrics=["accuracy_score"], show_oracle=False, samplong_rate = 5):
        """
        Plot evaluation for the prediction and its transition when human can check the prediction from samples with higher penalty to samples with lower penalty.
        When all samples are checked, which means no cost-cut is expected, the evaluation reach 1.

        Parameters
        __________
        label_array : 1d array of true class label

        metrics : list, default = ["accuracy_score"]
            You can choose how to evaluate the prediction; accuracy_score, f1_score, precision_score, recall_score.

        show_oracle : Bool
            If you know all anser classes for each sample, you can check only false prediction and reach 1 with the minimum human cost. This is called
            oracle. It is not realistic though, it is useful for comparison.

        samplong_rate : int or float (0,100], default = 5
            The smaller the sampling rate, the smoother transition line you will get.
        
        """
        assert type(metrics) == list
        self._sampling_rate = samplong_rate
        self.n_sampling = int(100 / samplong_rate + 1)
        self.df["ans"] = label_array
        self.emerging_label = sorted(list(set(label_array))) #全てのラベルが必ずしも正解ラベルにあるわけではない
        sorted_df = self.df.sort_values("mu_min", ascending=False).reset_index()
        self.scores = self._calc_scores(sorted_df, metrics)

        n_plot = len(metrics)
        x_axis = np.linspace(0,100,self.n_sampling)
        plt.figure(figsize=(6,5*n_plot))
        for i,x in enumerate(metrics):
            plt.subplot(n_plot,1,i+1)
            for j,y in enumerate(self.emerging_label): #各クラスを陽性ラベルとしたときのラインをかく
                if x == "accuracy_score":
                    plt.plot(x_axis, self.scores[x], label = "label" + str(y)) 
                    if show_oracle:
                        plt.plot(x_axis, self._calc_score_oracle(self.df, x, y), label = "oracle_" + "label" + str(y)) 
                    self.check_rate_df[f"{x}(label{y})"] = self.check_rates(self.scores[x])
                else:
                    plt.plot(x_axis, self.scores[x][:,j], label = "label" + str(y)) 
                    if show_oracle:
                        plt.plot(x_axis, self._calc_score_oracle(self.df, x, y)[:,j], label = "oracle_" + "label" + str(y)) 
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
        score =  score.tolist() #len(score) = len(self.emerging_label), type(score) = np.ndarray 
        return score 
            
    def _calc_scores(self, sorted_df, metrics)->dict:
        scores_dict = {}
        for x_score in metrics:
            scores_dict[x_score] = np.array([self._calc_score(n, sorted_df, score_type=x_score) for n in np.linspace(0,sorted_df.shape[0], self.n_sampling)]) #shape は(データ数, num_class)
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
        oracle_score = [self._calc_score(n, df_oraclesort, score_type) for n in np.linspace(0,df_oraclesort.shape[0], self.n_sampling)]
        oracle_score = np.array(oracle_score)

        return oracle_score #[[...],[...],...,[...]] #各行のk番目にラベルkの時のスコアが入っている

    def check_rate(self, threshold, score_array):
        score_list = list(score_array)
        score = score_array[score_array >= threshold][0] #初めて閾値を超えた時のスコア
        check_rate = score_list.index(score)
        return check_rate * self._sampling_rate

    def check_rates(self, score_array):
        thresholds_list = [0.90, 0.95, 0.99]
        check_rates = []
        for threshold in thresholds_list:
            check_rates.append(self.check_rate(threshold, score_array))
        return check_rates
        