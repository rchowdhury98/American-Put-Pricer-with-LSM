#include <iostream>
#include <string>
#include <random>
#include <iomanip>
#include <Eigen/Dense>
 
using namespace Eigen;
using namespace std;

VectorXd generateFinalPaths(float r, float sigma, int M, float T, mt19937 generator)
{
    double mean = (r - 0.5 * pow(sigma, 2.0)) * T;
    double stddev = sqrt(T) * sigma;

    normal_distribution<double> distribution(mean, stddev);

    VectorXd paths(M);

    for(int i =0; i < M; i++)
    {
        paths(i) = distribution(generator);
    }

    return paths;
}

VectorXd generateFinalValues(VectorXd paths, double strike_price, double stock_price)
{
    VectorXd exerciseValues =  (VectorXd::Constant(paths.size(), 1.0) * strike_price).array() - ((paths.array().exp()) * stock_price);
    VectorXd zeroVector = VectorXd::Constant(paths.rows(), 0.0);

    return zeroVector.array().max(exerciseValues.array());
}

MatrixXd getPredictorFromPaths(double stock_price, VectorXd paths, int degree)
{
    MatrixXd pricePredictor = MatrixXd(paths.rows(), degree + 1);

    for (int d = 0; d < degree + 1; d ++)
    {
        pricePredictor.col(d) = (paths.array().exp() * stock_price).array().pow(d);
    }

    return pricePredictor;
}

//TODO - Fix newtons raphsons issue
double solveHoldEqualsExercise(VectorXd linearFitCoefs, double strike_price, double initialGuess, int max_descent_interations)
{
    // cout << "Coefficients: " << linearFitCoefs << endl;
    double prevValue = initialGuess;
    double currentValue = initialGuess;
    double stepDiff = 1.0;
    int i = 0;

    // cout << "Relative difference: " << abs(stepDiff) / max(currentValue, max(prevValue, 1.0)) << endl;

    while (abs(stepDiff) / max(currentValue, max(prevValue, 1.0)) > 0.00001)
    {
        double f_x = currentValue - strike_price;

        // cout << "f(x) <- " << f_x << endl;

        for (int a = 0; a < linearFitCoefs.size(); a++)
        {
            // cout << "f(x) += " << linearFitCoefs(a) * pow(currentValue, a) << endl;
            f_x += linearFitCoefs(a) * pow(currentValue, a);
        }

        // cout << "f(x) = " << f_x << endl;

        double f_p_x = 1.0;

        for (int a = 1; a < linearFitCoefs.size(); a++)
        {
            // cout << "f\'(x) = " << linearFitCoefs(a) * a * pow(currentValue, a-1) << endl;
            f_p_x += linearFitCoefs(a) * a * pow(currentValue, a-1);
        } 
        // cout << "f\'(x) = " << f_p_x << endl;
        prevValue = currentValue;
        currentValue -= (f_x / f_p_x);
        // cout << " x(i) = " << currentValue << endl;

        stepDiff = currentValue - prevValue;
        // cout << "Difference: " << stepDiff << endl;
        // cout << "Relative difference: " << abs(stepDiff) / max(currentValue, max(prevValue, 1.0)) << "\n" << endl;

        i++;
        if (i == max_descent_interations)
        {
            return currentValue;
        }
    }

}

double solveHoldEqualsZero(VectorXd linearFitCoefs, double strike_price, double initialGuess)
{
    double prevValue = initialGuess;
    double currentValue = initialGuess;
    double stepDiff = 1.0;
    int i = 0;

    while (abs(stepDiff) / max(currentValue, max(prevValue, 1.0)) > 0.0001)
    {
        double f_x = 0.0;

        for (int a = 0; a < linearFitCoefs.size(); a++)
        {
            f_x += linearFitCoefs(a) * pow(currentValue, double(a));
        } 

        double f_p_x = 0.0;

        for (int a = 1; a < linearFitCoefs.size(); a++)
        {
            f_p_x += linearFitCoefs(a) * double(a) * pow(currentValue, double(a-1));
        } 
        prevValue = currentValue;
        currentValue -= (f_x / f_p_x);
        double diff = currentValue - prevValue;

        i++;
        if (i == 1000)
        {
            return currentValue;
        }
    }

}

int main(int argc, char* argv[]) {
    if(argc!=10)
    {
        cout << "Insufficient args!" << endl;
        return 1;
    }

    const double sigma = stod(argv[1]);               // real world volatility
    const double stock_price = stod(argv[2]);        // starting security price
    const double strike_price = stod(argv[3]);       // strike price
    const double r = stod(argv[4]);                   // risk free return rate
    const double delta_t = stod(argv[5]);             // interval for timesteps
    const int n = stoi(argv[6]);                     // timesteps
    const int M = stoi(argv[7]);                     // # of Monte Carlo paths to generate
    const int degree = stoi(argv[8]);                // degree of polynomial to fit to
    const int max_descent_interations = stoi(argv[9]);

    const double T = double(n)*delta_t;                       //Calculate termination time for the contract

    random_device rd;
    mt19937 generator(rd());

    cout << "Generating first Monte Carlo Final Values" << endl;
    VectorXd paths = generateFinalPaths(r, sigma, M, T, generator);
    VectorXd optimalThreshold = VectorXd::Constant(n + 1, 1.0) * strike_price;

    VectorXd values = generateFinalValues(paths, strike_price, stock_price);

    cout << "First set of Monte Carlo Sims" << endl;
    for(int i = n-1; i > 0; i--)
    {
        for(int j = 0; j < M; j++)
        {
            double mean = paths(j)*(1.0-(1.0/double(i+1.0)));
            double stddev = sqrt(delta_t*(1.0-(1.0/double(i+1.0)))) * sigma;
            normal_distribution<double> distribution(mean, stddev);

            paths(j) = distribution(generator);
            values(j) = exp(-r*delta_t)*values(j);
        }


        MatrixXd pricePredictor = getPredictorFromPaths(stock_price, paths, degree);
        MatrixXd predictorPseudoInverse = pricePredictor.completeOrthogonalDecomposition().pseudoInverse();  // TODO - Look into the decomp method
        VectorXd valuesT = values;

        cout << pricePredictor << endl;
        cout << predictorPseudoInverse << endl;
        cout << valuesT << endl;
        VectorXd linearFitCoefs = predictorPseudoInverse * valuesT;
        cout << linearFitCoefs << endl;

        // return 1;
        double exerciseSolution = solveHoldEqualsExercise(linearFitCoefs, strike_price, optimalThreshold(i+1), max_descent_interations);

        if (exerciseSolution < 0.001 || exerciseSolution > strike_price)
        {
            // exerciseSolution = solveHoldEqualsZero(linearFitCoefs, strike_price, optimalThreshold(i+1));

            // if (exerciseSolution < 0.001 || exerciseSolution > strike_price)
            // {
                exerciseSolution = optimalThreshold(i + 1);
            // }
        }

        optimalThreshold(i) = exerciseSolution;

        for(int j = 0; j < M; j++)
        {
            if(stock_price * exp(paths(j)) <= optimalThreshold(i))
            {
                values(j) = max(0.0, strike_price - stock_price*exp(paths(j)));
            }
        }

        if(i%100 == 0)
        {
            cout << i << " steps left" << endl;
            cout << "current value of contract: " << values.mean() << endl;
        }
    }

    optimalThreshold(0) = exp(-r*delta_t)*optimalThreshold(1);

    cout << "Generating second Monte Carlo Final Values" << endl;
    paths = generateFinalPaths(r, sigma, M, T, generator);
    values = generateFinalValues(paths, strike_price, stock_price);


    cout << "Second set of Monte Carlo Sims" << endl;
    for(int i = n-1; i > 0; i--)
    {
        for(int j = 0; j < M; j++)
        {
            double mean = paths(j)*(1.0-(1.0/double(i+1)));
            double stddev = sqrt(delta_t*(1.0-(1.0/double(i+1)))) * sigma;
            normal_distribution<double> distribution(mean, stddev);

            paths(j) = distribution(generator);
            values(j) = exp(-r*delta_t)*values(j);

            if(stock_price * exp(paths(j)) <= optimalThreshold(i))
            {
                values(j) = max(0.0, strike_price - stock_price*exp(paths(j)));
            }
        }

        if(i%100 == 0)
        {
            cout << i << " steps left" << endl;
            cout << "current value of contract: " << values.mean() << endl;
        }
    }

    double valueHold = values.mean();

    cout << max(max(0.0, strike_price - stock_price), valueHold) << endl;

    return 0;
}
