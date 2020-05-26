import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score
import copy


class Cooperante(object):
    def __init__(self, penalty_matrix, class_to_check=[]):
        """
        Parameters
        ----------
        penalty_matrix : ndarray of shape (n_classes, n_classes)
            Penalty_matrix[i, j] means that when class i is predicted as class j, the penalty will be added. You can avoid some mis-predicting
            by setting the penalty high. Penalty_matrix[i, i] should be 0.
        
        class_to_check : list, default=[]
            When you want to check every sample predicted as some class, the class number should be put in here. This module will return
            predicted class except the class specified here.

        Notes
        -----
        
        Example
        _______
        penalty matrix
          a b c d
        a 0 1 1 1
        b 9 0 1 1
        c 9 1 0 1
        d 1 1 1 0


        """
        
        assert penalty_matrix.shape[0] == penalty_matrix.shape[1]
        self.penalty_matrix = penalty_matrix
        self.n_classes = self.penalty_matrix.shape[1]
        self.class_to_check = class_to_check if type(class_to_check) == list else [class_to_check]

    def fit(self, probability_array)->np.array:
        """
        Parameters
        ----------
        probability_array : ndarray of shape (n_samples, n_classes)
            The output from some classifier should be here.
            example:
            a b c d
            0.1 0.2 0.3 0.4
            0.7 0.1 0.1 0.1

       
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
        assert min(probability_array) >= 0.
        assert max(probability_array) <= 1.

        # １列で与えられたときの処理
        if len(probability_array.shape) == 1:
            
            probability_array = probability_array[:,np.newaxis]
            probability_array = np.concatenate([1 - probability_array, probability_array], axis=1)
            
        assert probability_array.shape[1] == self.penalty_matrix.shape[1]

        mu_for_each_label = np.dot(probability_array, self.penalty_matrix).astype(float) #内積をとる #あとでnanを入れたいのでfloatにしておく
        mu_for_each_label[:, self.class_to_check] = np.nan  #指定したラベルの列にnanを挿入
        mu_min = np.nanmin(mu_for_each_label, axis=1) #最小のペナルティー期待値
        mu_min_index = np.nanargmin(mu_for_each_label, axis=1) #これが選ぶラベル。ただしclass_to_checkで指定された列を抜いていることに注意        
        prediction = mu_min_index.astype(int) #予測ラベルがfloatになってるのでintに戻す
        self.prediction = prediction
        self.mu_min = mu_min

        return mu_min, prediction

    def plot_eval(self, label_array, metrics="accuracy_score", class_ref=1, show_oracle=False, sampling_rate = 5):
        """
        Plot evaluation for the prediction and its transition when human can check the prediction from samples with higher penalty to samples with lower penalty.
        When all samples are checked, which means no cost-cut is expected, the evaluation reach 1.

        Parameters
        __________
        label_array : 1d array of true class label

        metrics : str, default = "accuracy_score"
            You can choose how to evaluate the prediction; accuracy_score, f1_score, precision_score, recall_score.

        class_ref : a class or list of classes
            You can specify the class used to calcurate the score.

        show_oracle : Bool
            If you know all anser classes for each sample, you can check only false prediction and reach 1 with the minimum human cost. This is called
            oracle. It is not realistic though, it is useful for comparison.

        samplong_rate : int or float (0,100], default = 5
            The smaller the sampling rate, the smoother transition line you will get.
        
        """
        self._sampling_rate = sampling_rate
        self.n_sampling = int(100 / sampling_rate + 1)
        self.df = pd.DataFrame(self.prediction, columns=["pred"])  
        self.df["mu_min"] = self.mu_min
        self.df["ans"] = label_array.astype(int)
        class_ref = class_ref if type(class_ref) == list else [class_ref] #class_ref should be list
        class_ref = sorted(class_ref) 
        sorted_df = self.df.sort_values("mu_min", ascending=False).reset_index()
        self.scores = self._calc_scores(sorted_df, metrics, label=class_ref)
        self.check_rate_df = pd.DataFrame([90, 95, 99], columns = ["score over X %"])

        x_axis = np.linspace(0,100,self.n_sampling)
        for j,y in enumerate(class_ref): #各クラスを陽性ラベルとしたときのラインをかく
            if metrics == "accuracy_score":
                plt.plot(x_axis, self.scores[metrics], label = "label" + str(y)) 
                if show_oracle:
                    plt.plot(x_axis, self._calc_score_oracle(self.df, metrics, y), label = "oracle_" + "label" + str(y)) 
                self.check_rate_df[f"{metrics}(label{y})"] = self.check_rates(self.scores[metrics])
            else:
                plt.plot(x_axis, self.scores[metrics][:,j], label = "label" + str(y)) 
                if show_oracle:
                    plt.plot(x_axis, self._calc_score_oracle(self.df, metrics, y), label = "oracle_" + "label" + str(y)) 
                self.check_rate_df[f"{metrics}(label{y})"] = self.check_rates(self.scores[metrics][:,j])
                
        if len(class_ref) >= 2:
            plt.legend()
        plt.xlabel('Human Check Percent')
        plt.xlim(0,100)
        plt.grid()
        plt.show()

    def _threshhold_check(self, label_array, metrics="accuracy_score", class_ref=1, show_oracle=False, sampling_rate = 5):

        pass

    def _calc_score(self, n, df, score_type = "f1_score", label=[1])->list: #dataframeのn番目までhuman checkしてスコアを返す  
        assert type(label) == list
        n = int(n)
        dataframe = df.copy()
        
        dataframe.loc[:n, "pred"] =dataframe.loc[:n, "ans"]
        if score_type == "accuracy_score":
            score = accuracy_score(dataframe["ans"],dataframe["pred"])
        elif score_type == "f1_score":
            score = list(f1_score(dataframe["ans"],dataframe["pred"], average = None, labels=label))
        elif score_type == "precision_score":
            score = list(precision_score(dataframe["ans"],dataframe["pred"],  average = None, labels=label))
        elif score_type == "recall_score":
            score = list(recall_score(dataframe["ans"],dataframe["pred"],  average = None, labels=label))
        else:
            raise ValueError(score_type)
        return score
        
    def _calc_scores(self, sorted_df, metrics, label)->dict:
        scores_dict = {}
        scores_dict[metrics] = np.array([self._calc_score(n, sorted_df, metrics, label) for n in np.linspace(0,sorted_df.shape[0], self.n_sampling)])
        return scores_dict

    def _sort_df_oracle(self, df, metrics = "f1_score", label = 1)->pd.DataFrame:
        #正解ラベル、予測ラベルのついたdf
        #sort
        assert type(label) == int
        pred_negative = df[df["pred"] == label]
        ans_negative = df[df["ans"] == label]

        if metrics == "precision_score":
            wrong_idx = pred_negative[pred_negative["ans"] != pred_negative["pred"]].index
            else_idx = df.drop(index = wrong_idx).index

        elif metrics == "recall_score":
            wrong_idx = ans_negative[ans_negative["ans"] != ans_negative["pred"]].index
            else_idx = df.drop(index = wrong_idx).index

        elif metrics  == "f1_score":
            wrong_idx_precision = pred_negative[pred_negative["ans"] != pred_negative["pred"]].index
            wrong_idx_recall = ans_negative[ans_negative["ans"] != ans_negative["pred"]].index
            wrong_idx = wrong_idx_precision.append(wrong_idx_recall)
            else_idx = df.drop(index = wrong_idx).index

        elif metrics == "accuracy_score":
            wrong_idx_precision = pred_negative[pred_negative["ans"] != pred_negative["pred"]].index
            wrong_idx_recall = ans_negative[ans_negative["ans"] != ans_negative["pred"]].index
            wrong_idx = wrong_idx_precision.append(wrong_idx_recall)
            else_idx = df.drop(index = wrong_idx).index

        oracle_idx = wrong_idx.append(else_idx)
        sorted_df = df.loc[oracle_idx, :].reset_index()  #間違ってるものを上に持ってきた
        return sorted_df

    def _calc_score_oracle(self, df, metrics = "f1_score", label = 1)->np.array:
        assert type(label) == int
        sorted_df = self._sort_df_oracle(df, metrics, label)
        oracle_score = [self._calc_score(n, sorted_df, metrics, [label]) for n in np.linspace(0,sorted_df.shape[0], self.n_sampling)]
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
        