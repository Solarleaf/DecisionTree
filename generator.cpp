// generator.cpp
// filesystem requires C++17

#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <filesystem>

// Shopping session
struct Session
{
    int administrative;      // Account Settings
    int product;             // Pages with products
    int information;         // Information pages
    double bounceRate;       // Percent that leave after one page
    double exitRate;         // Percent that
    double pageValue;        // Page usua value of a page
    std::string visitorType; // Returning, new, or cookies disabled
    bool weekend;            // True if weekend. False if Weekday
    bool purchase;           // Did they make a purchase
};

std::vector<Session> generateSessions(int n)
{
    std::vector<Session> data;
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<> adminDist(0, 5);
    std::uniform_int_distribution<> prodDist(0, 20);
    std::uniform_int_distribution<> infoDist(0, 10);
    std::uniform_real_distribution<> rateDist(0.0, 1.0);
    std::uniform_real_distribution<> dist(0.0, 1.0);
    std::exponential_distribution<> pageValDist(0.1);
    std::bernoulli_distribution weekendDist(0.3);
    std::bernoulli_distribution visitorDist(0.7); // 70% returning

    for (int i = 0; i < n; ++i)
    {
        Session s;
        s.administrative = adminDist(gen);
        s.product = prodDist(gen);
        s.information = infoDist(gen);
        s.bounceRate = rateDist(gen);
        s.exitRate = rateDist(gen);
        s.pageValue = pageValDist(gen) * 5;
        s.visitorType = visitorDist(gen) ? "Returning_Visitor" : "New_Visitor";
        s.weekend = weekendDist(gen);

        // Start with a base probability
        double purchaseProb = 0.0;

        // Feature Influence
        purchaseProb += s.pageValue / 400.0;  // Higher page value helps
        purchaseProb += s.product * 0.01;     // More product pages helps
        purchaseProb += s.information * 0.01; // Info helps slightly
        purchaseProb -= s.bounceRate * 0.3;   // High bounce hurts
        purchaseProb -= s.exitRate * 0.2;     // High exit hurts
        purchaseProb += (s.visitorType == "Returning_Visitor") ? 0.30 : 0.0;
        purchaseProb += s.weekend ? 0.05 : 0.0; // Slight weekend boost

        // Keeping it between 0 and 1
        purchaseProb = std::min(purchaseProb, 1.0);
        purchaseProb = std::max(purchaseProb, 0.0);

        // Inject randomness
        std::bernoulli_distribution rareWin(0.02);  // 2% chance of random yes
        std::bernoulli_distribution rareFail(0.05); // 5% chance of random no
        std::bernoulli_distribution purchaseDist(purchaseProb);

        // This is to randomly have it be yes or randomly be no
        double r = dist(gen);
        if (r < 0.02)
            s.purchase = true; // 2% random yes
        else if (r > 0.95)
            s.purchase = false; // 5% random no
        else
            s.purchase = (r < purchaseProb);

        data.push_back(s);
    }
    return data;
}

void writeSessionsToCSV(const std::vector<Session> &data, const std::string &filename)
{
    std::ofstream file(filename);
    file << "Administrative,Product,Information,BounceRate,ExitRate,PageValue,VisitorType,Weekend,Purchase\n";
    for (const auto &s : data)
    {
        file << s.administrative << "," << s.product << "," << s.information << "," << s.bounceRate << ","
             << s.exitRate << "," << s.pageValue << "," << s.visitorType << "," << s.weekend << "," << s.purchase << "\n";
    }
}

int main()
{
    std::vector<Session> trainingData = generateSessions(1000);
    std::vector<Session> actualData = generateSessions(300);

    std::string output = "Data_Input/";
    std::filesystem::create_directories(output);
    // Training Data
    writeSessionsToCSV(trainingData, output + "shoppers_train.csv");
    // Actual Results
    writeSessionsToCSV(actualData, output + "shoppers_actual.csv");
    return 0;
}
