import numpy as np
import pandas as pd


class MLOptimizer(object):
    """
    Alternating Direction Method of Multipliers + Cyclic Coordinate Descent
    Can only be applied to weights greater than 0
    To calculate short weights, create another instance of this class
    Use method:
    1. Create an instance of this class
        opt = MLOptimizer(df_signal, df_today, coin_list, long)
    2. Call sigma_cal function to calculate sigma matrix
        opt.sigma_cal()
    3. Call optimize function to calculate optimal weights
        opt.optimize()
    """
    def __init__(self, df_signal, df_today, coin_list, long: int):
        """
        :param df_signal: Historical data df
        :param df_today: Data of today
        :param coin_list: Target coins
        :param long: 1 for long, -1 for short
        """
        self.df_signal = df_signal
        self.df_today = df_today

        # Must standardize signal
        self.df_today.loc[:, 'signal'] = ((self.df_today['signal'] - self.df_today['signal'].mean())
                                          / self.df_today['signal'].std())
        self.df_today.loc[:, 'signal'] = self.df_today['signal'].apply(lambda x: x if long > 0 else -x)
        self.coin_list = coin_list

        # Do not use equal weights
        self.w = np.zeros((len(coin_list), 1)) / len(coin_list)

        # Initialize sigma matrix
        self.sigma_mat = pd.DataFrame()

    def __signal_data(self):
        """
        Get historical signal data
        :return: Historical signal data, one column for each coin
        """
        end_date = self.df_today['date'].iloc[0]

        data_list = []

        for coin in self.coin_list:
            df_coin = self.df_signal[self.df_signal['token'] == coin]
            df_coin = df_coin[df_coin['date'] < end_date]
            df_coin.sort_values(by='date', inplace=True)
            df_coin.reset_index(drop=True, inplace=True)

            data_list.append(df_coin[['signal']].copy())

        signal_df = pd.concat(data_list, axis=1)

        return signal_df

    def sigma_cal(self):
        """
        Calculate the covariance matrix of the historical signal data
        :return: Covariance matrix, n * n with n equal to the number of coins
        """
        self.sigma_mat = self.__signal_data().cov()

    def __update_x(self, w, lam, i, y, u, c=0.5):
        """
        Update the x_i in the CCD algorithm
        :param w: Current portfolio weights
        :param lam: Coefficient of the logarithmic weighted sum term
         A greater lam means a more significant role of the signal
        :param i: Direction
        :param c: Risk averse coefficient in Risk Budget Model
        :return: Updated x_i
        """
        w_new = w.copy()
        sigma_mat = self.sigma_mat

        sigma_x = np.sqrt(np.matmul(w.T, np.matmul(sigma_mat, w)))
        sigma_x = sigma_x.iloc[0, 0]

        RB_i = (0 if self.df_today['signal'].iloc[i] <= 0
                else self.df_today['signal'].iloc[i] / np.sum(self.df_today[self.df_today['signal'] > 0]['signal']))

        alpha_i = c * sigma_mat.iloc[i, i] + sigma_x

        beta_i = (c * (sigma_mat.iloc[i, :].sum() - sigma_mat.iloc[i, i])
                - self.df_today['signal'].iloc[i] * sigma_x - (y - u) * sigma_x)

        gamma_i = - lam * RB_i * sigma_x

        # Get new weight by solving the quadratic equation
        new_value = (-beta_i + np.sqrt(beta_i ** 2 - 4 * alpha_i * gamma_i)) / (2 * alpha_i)

        w_new[i, 0] = new_value

        return w_new

    def __ccd_algo(self, w, lam, y, u):
        """
        CCD algorithm
        :param w: Initial weights
        :param lam: Coefficient of the logarithmic weighted sum term
         A greater lam means a more significant role of the signal
        :param y: Previous y in ADMM，used for updating x
        :param u: Previous u in ADMM，used for updating x
        :return: Updated weights
        """
        n = len(w)  # token nums
        w_prev = 1
        max_iter = 50

        while np.linalg.norm(w - w_prev) > 1e-3 and max_iter > 0:
            for i in range(n):
                w_prev = w
                w = self.__update_x(w, lam, i, y[i, 0], u[i, 0])
                w = np.nan_to_num(w, nan=0.0)  # avoid nan values

            max_iter -= 1

        return w

    def __project(self, w):
        """
        Projection function
        :param w: Weights to be projected
        :return: Projected weights
        """
        w_base = 1 / len(w)
        w = np.where(w > 1.1 * w_base, 1.1 * w_base, w)
        w = np.where(w < 0.9 * w_base, 0.9 * w_base, w)

        return w

    def __admm_algo(self, w, lam):
        """
        ADMM algorithm
        Updates x, y, and u iteratively
        x is updated by the CCD algorithm
        y is updated by projection
        Punishment phi is selected to be 1
        :param w: Initial weights. Not for use, just for shape
        :param lam: Coefficient of the logarithmic weighted sum term
         A greater lam means a more significant role of the signal
        :return: Final weights
        """
        x = np.ones(w.shape) / len(w)
        y = np.ones(w.shape) / len(w)
        u = np.zeros(w.shape)
        y_prev = np.ones(w.shape)
        max_iter = 50

        while (np.linalg.norm(y - y_prev) > 1e-3 or np.linalg.norm(x - y) > 1e-3) and max_iter > 0:
            y_prev = y
            x = self.__ccd_algo(x, lam, y, u)
            y = self.__project(x + u)
            u = u + x - y
            max_iter -= 1

        return y

    def optimize(self):
        """
        Optimization function:
        First find two lambdas among which one makes the sum of weights greater than 1,
         and the other makes the sum of weights less than 1.
        Iterate until the sum of weights converges to 1.
        :return: Optimized weights
        """
        lambda_min = 0
        lambda_max = 20
        w = np.ones(self.w.shape) / len(self.w)
        max_iter = 50

        # Iterate until the sum of weights is equal to 1
        while np.abs(np.sum(self.w) - 1) > 1e-3 and max_iter > 0:

            lambda_new = (lambda_max + lambda_min) / 2
            self.w = self.__admm_algo(w, lambda_new)

            if np.sum(self.w) < 1:
                lambda_min = lambda_new

            elif np.sum(self.w) > 1:
                lambda_max = lambda_new

            max_iter -= 1

        # Tune the weights to sum up to 1
        sum_weight = np.sum(self.w)
        ratio = sum_weight / 1
        self.w /= ratio


