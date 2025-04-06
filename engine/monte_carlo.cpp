#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <omp.h>

namespace py = pybind11;


// Einfache Monte-Carlo-Simulation für eine geometrische Brownsche Bewegung
py::array_t<double> monte_carlo_gbm(int num_paths, int num_steps, double S0, double mu, double sigma, double dt) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    auto result = py::array_t<double>({num_paths, num_steps});
    auto r = result.mutable_unchecked<2>();

    #pragma omp parallel for
    for (int i = 0; i < num_paths; ++i) {
        double S = S0;
        for (int j = 0; j < num_steps; ++j) {
            S *= exp((mu - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * d(gen));
            r(i, j) = S;
        }
    }

    return result;
}


py::array_t<double> monte_carlo_jump_diffusion(int num_paths, int num_steps, double S0, double mu, double sigma, double mu_j, double sigma_j, double lambda, double dt) {
    std::poisson_distribution<int> poisson_dist(lambda*dt);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    auto result = py::array_t<double>({num_paths, num_steps});
    auto r = result.mutable_unchecked<2>();
    #pragma omp parallel for
    for (int i = 0; i < num_paths; ++i) {
        double S = S0;
        for (int j = 0; j < num_steps; ++j) {
            S *= exp((mu - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * d(gen));
            if(poisson_dist(gen)==1){
                S *= exp(d(gen) * sigma_j + mu_j);
            }
            r(i, j) = S;
        }
    }
    return result;
}

py::array_t<double> monte_carlo_heston(int num_paths, int num_steps, double S0, double V0, double mu, double kappa, double theta, double xi, double rho, double dt) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    double Z_s;
    double Z_v;

    auto result = py::array_t<double>({num_paths, num_steps});
    auto r = result.mutable_unchecked<2>();
    #pragma omp parallel for
    for (int i = 0; i < num_paths; ++i) {
        double S = S0;
        double V = V0;
        for (int j = 0; j < num_steps; ++j) {
            Z_s = d(gen);
            Z_v = rho * Z_s + sqrt(1 - rho * rho) * d(gen);
            S *= exp((mu - 0.5 * V) * dt + sqrt(V) * sqrt(dt) * Z_s);
            V = V + kappa * (theta - V) * dt + xi * sqrt(V * dt) * Z_v;
            r(i, j) = S;
        }
    }
    return result;
}

py::array_t<double> monte_carlo_bates(int num_paths, int num_steps, double S0, double V0, double mu, double kappa, double theta, double xi, double rho, double mu_j, double sigma_j, double lambda, double dt) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    std::poisson_distribution<int> poisson_dist(lambda*dt);
    double Z_s;
    double Z_v;
    auto result = py::array_t<double>({num_paths, num_steps});
    auto r = result.mutable_unchecked<2>();
    #pragma omp parallel for
    for (int i = 0; i < num_paths; ++i) {
        double S = S0;
        double V = V0;
        for (int j = 0; j < num_steps; ++j) {
            Z_s = d(gen);
            Z_v = rho * Z_s + sqrt(1 - rho * rho) * d(gen);
            S *= exp((mu - 0.5 * V) * dt + sqrt(V) * sqrt(dt) * Z_s);
            if(poisson_dist(gen)==1){
                S *= exp(d(gen) * sigma_j + mu_j);
            }
            V = V + kappa * (theta -V) * dt + xi * sqrt(V * dt) * Z_v;
            r(i, j) = S;
        }
    }
    return result;
}

py::array_t<double> monte_carlo_varaince_gamma(int num_paths, int num_steps, double S0, double mu, double sigma, double gamma, double alpha, double beta, double dt) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    std::gamma_distribution<double> g(alpha,beta);

    auto result = py::array_t<double>({num_paths, num_steps});
    auto r = result.mutable_unchecked<2>();
    #pragma omp parallel for
    for (int i = 0; i < num_paths; ++i) {
        double S = S0;
        for (int j = 0; j < num_steps; ++j) {
            S *= exp((mu - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * d(gen) + gamma * sqrt(dt) * g(gen));
            r(i, j) = S;
    }
}
    return result;
}


py::array_t<double> monte_carlo_vasicek(int num_paths, int num_steps, double r0, double theta, double sigma, double kappa, double dt) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    auto result = py::array_t<double>({num_paths, num_steps});
    auto r = result.mutable_unchecked<2>();
    #pragma omp parallel for
    for (int i = 0; i < num_paths; ++i) {
        double R = r0;
        for (int j = 0; j < num_steps; ++j) {
            R += kappa * (theta - R) * dt + sigma * sqrt(dt) * d(gen);
            r(i,j) = R;
    }
}
    return result;
}


py::array_t<double> monte_carlo_cir(int num_paths, int num_steps, double r0, double theta, double sigma, double kappa, double dt) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    auto result = py::array_t<double>({num_paths, num_steps});
    auto r = result.mutable_unchecked<2>();
    #pragma omp parallel for
    for (int i = 0; i < num_paths; ++i) {
        double R = r0;
        for (int j = 0; j < num_steps; ++j) {
            R += kappa * (theta - R) * dt + sigma * sqrt(R) *sqrt(dt) * d(gen);
            r(i,j) = R;
    }
}
    return result;
}

PYBIND11_MODULE(monte_carlo, m) {
    m.def("gbm", &monte_carlo_gbm, "Monte Carlo Simulation for GBM",
          py::arg("num_paths"), py::arg("num_steps"), py::arg("S0"),
          py::arg("mu"), py::arg("sigma"), py::arg("dt"));
    m.def("jump_diffusion", &monte_carlo_jump_diffusion, "Monte Carlo Simulation for Merton Jump Diffusion",
        py::arg("num_paths"), py::arg("num_steps"), py::arg("S0"),
        py::arg("mu"), py::arg("sigma"),py::arg("mu_j"), py::arg("sigma_j"),py::arg("lambda"), py::arg("dt"));
    m.def("heston", &monte_carlo_heston, "Monte Carlo Simulation for Heston process using Euler-Maruyama",
        py::arg("num_paths"), py::arg("num_steps"), py::arg("S0"),py::arg("V0"),
        py::arg("mu"), py::arg("kappa"),py::arg("theta"), py::arg("xi"),py::arg("rho"), py::arg("dt"));
    m.def("bates", &monte_carlo_bates, "Monte Carlo Simulation for Bates process",
        py::arg("num_paths"), py::arg("num_steps"), py::arg("S0"),py::arg("V0"),
        py::arg("mu"), py::arg("kappa"),py::arg("theta"), py::arg("xi"),py::arg("rho"),py::arg("mu_j"), py::arg("sigma_j"), py::arg("lambda"),py::arg("dt"));
    m.def("variance_gamma", &monte_carlo_varaince_gamma, "Monte Carlo Simulation for Variance-Gamma process",
        py::arg("num_paths"), py::arg("num_steps"), py::arg("S0"),py::arg("mu"), py::arg("sigma"), py::arg("gamma"),py::arg("alpha"), py::arg("beta"), py::arg("dt"));
    m.def("vasicek", &monte_carlo_vasicek, "Monte Carlo Simulation for Vasicek process",
        py::arg("num_paths"), py::arg("num_steps"), py::arg("r0"),
        py::arg("thata"), py::arg("sigma"), py::arg("kappa"), py::arg("dt"));
    m.def("cir", &monte_carlo_cir, "Monte Carlo Simulation for Cox-Ingersoll-Ross process",
        py::arg("num_paths"), py::arg("num_steps"), py::arg("r0"),
        py::arg("thata"), py::arg("sigma"), py::arg("kappa"), py::arg("dt"));
        }
