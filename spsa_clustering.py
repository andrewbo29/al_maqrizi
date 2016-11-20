import numpy as np
from scipy.spatial.distance import euclidean, mahalanobis


class Noise(object):
    """Noise object"""

    def __init__(self, func=None, name=None):
        self.func = func
        self.name = name

    def fabric(self, x):
        return self.func(x)

    def __repr__(self):
        return self.name


class ClusteringSPSA(object):
    """Gaussian mixture model SPSA clustering"""

    def __init__(self, n_clusters, Gammas=None, alpha=lambda x: 0.001, beta=lambda x: 0.001, norm_init=False,
                 noise=Noise(func=lambda x: 0, name='0')):
        self.n_clusters = n_clusters
        self.Gammas = Gammas
        self.labels_ = np.zeros(0)
        self.cluster_centers_ = []
        self.alpha = alpha
        self.beta = beta
        self.norm_init = norm_init
        self.noise = noise
        self.cluster_centers_list = []
        self.iteration_num = 1

    def fit(self, w):
        if self.iteration_num % 100 == 0 or self.iteration_num == 1:
            print 'SPSA clustering iteration: %d' % self.iteration_num

        if self.Gammas is None:
            self.Gammas = [np.eye(w.shape[0]) for _ in xrange(self.n_clusters)]

        if self.norm_init:
            if self.iteration_num == 1:
                self.cluster_centers_ = np.random.multivariate_normal(np.zeros(w.shape[0]), np.eye(w.shape[0]),
                                                                      size=self.n_clusters)
                self.cluster_centers_list.append(self.cluster_centers_.copy())
            self.fit_step(w)
        else:
            if self.iteration_num <= self.n_clusters:
                self.cluster_centers_.append(w)
            else:
                if self.iteration_num == self.n_clusters + 1:
                    self.cluster_centers_ = np.array(self.cluster_centers_)
                    self.cluster_centers_list.append(self.cluster_centers_.copy())
                self.fit_step(w)

        self.iteration_num += 1

    def y_vec(self, centers, w):
        return np.array([mahalanobis(w, centers[label], np.linalg.inv(self.Gammas[label])) + self.noise.fabric(
                self.iteration_num) for label in xrange(self.n_clusters)])

    def j_vec(self, w):
        vec = np.zeros(self.n_clusters)
        vec[np.argmin(self.y_vec(self.cluster_centers_, w))] = 1
        return vec

    def delta_fabric(self, d):
        return np.where(np.random.binomial(1, 0.5, size=d) == 0, -1, 1)

    def alpha_fabric(self):
        return self.alpha(self.iteration_num)

    def beta_fabric(self):
        return self.beta(self.iteration_num)

    def fit_step(self, w):
        delta_n_t = self.delta_fabric(w.shape[0])[np.newaxis]
        alpha_n = self.alpha_fabric()
        beta_n = self.beta_fabric()

        j_vec = self.j_vec(w)[np.newaxis].T
        j_vec_dot_delta_t = np.dot(j_vec, delta_n_t)

        y_plus = self.y_vec(self.cluster_centers_ + beta_n * j_vec_dot_delta_t, w)[np.newaxis]

        y_minus = self.y_vec(self.cluster_centers_ - beta_n * j_vec_dot_delta_t, w)[np.newaxis]

        self.cluster_centers_ -= j_vec_dot_delta_t * np.dot(alpha_n * (y_plus - y_minus) / (2. * beta_n), j_vec)

        self.cluster_centers_list.append(self.cluster_centers_.copy())

    def cluster_decision(self, point):
        return np.argmin(
                [mahalanobis(self.cluster_centers_[label], point, np.linalg.inv(self.Gammas[label])) for label in
                 range(self.n_clusters)])

    def clusters_fill(self, data):
        self.labels_ = np.zeros(data.shape[0])
        for ind, point in enumerate(data):
            self.labels_[ind] = self.cluster_decision(point)


if __name__ == '__main__':
    N = 1000
    mix_prob = np.array([0.4, 0.4, 0.2])
    clust_means = np.array([[0, 0], [2, 2], [-2, 4]])
    clust_gammas = np.array([[[1, -0.7], [-0.7, 1]], np.eye(2), [[1, 0.7], [0.7, 1]]])
    data_set = []

    noise_0 = Noise(func=lambda x: 0, name='0')
    noise_1 = Noise(func=lambda x: np.random.normal(), name='$\mathcal{N}(0,1)$')
    noise_2 = Noise(func=lambda x: np.random.normal(0., 2.),
                    name='$\mathcal{N}(0,\sqrt{2})$')
    noise_3 = Noise(func=lambda x: np.random.normal(1., 1.),
                    name='$\mathcal{N}(1,1)$')
    noise_4 = Noise(func=lambda x: np.random.normal(1., 2.),
                    name='$\mathcal{N}(1,\sqrt{2})$')
    noise_5 = Noise(func=lambda x: 10 * (np.random.rand() * 4 - 2),
                    name='random')
    noise_6 = Noise(func=lambda x: 0.1 * np.sin(x) + 19 * np.sign(50 - x % 100),
                    name='irregular')
    noise_7 = Noise(func=lambda x: 20, name='constant')

    spsa_gamma = 1. / 6
    spsa_alpha = lambda x: 0.25 / (x ** spsa_gamma)
    spsa_beta = lambda x: 15. / (x ** (spsa_gamma / 4))
    # spsa_alpha = lambda x: 0.001
    # spsa_beta = lambda x: 0.001
    clustering = ClusteringSPSA(n_clusters=3, Gammas=None, alpha=spsa_alpha,
                                beta=spsa_beta, norm_init=False, noise=noise_0)

    true_labels = []

    sort_ind = np.argsort(np.linalg.norm(clust_means, axis=1))
    mix_prob = mix_prob[sort_ind]
    clust_means = clust_means[sort_ind]
    clust_gammas = clust_gammas[sort_ind]

    for _ in xrange(N):
        mix_ind = np.random.choice(len(mix_prob), p=mix_prob)
        data_point = np.random.multivariate_normal(clust_means[mix_ind],
                                                   clust_gammas[mix_ind])
        data_set.append(data_point)
        true_labels.append(mix_ind)
        clustering.fit(data_point)
    data_set = np.array(data_set)

    clustering.clusters_fill(data_set)
