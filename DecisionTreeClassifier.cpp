#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <numeric>
#include <random>
#include <algorithm>
#include <limits>
#include <memory>

using std::string;
using std::vector;

class TreeNode
{
public:
    bool isLeaf;
    int featureIndex;
    double threshold;
    int prediction;
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;

    TreeNode() : isLeaf(false), featureIndex(-1), threshold(0.0), prediction(-1), left(nullptr), right(nullptr) {}
};

class DecisionTreeClassifier
{
public:
    DecisionTreeClassifier(int depth, vector<string> names)
        : maxDepth(depth), featureNames(std::move(names)), actualMaxDepth(0) {}

    void fit(const vector<vector<double>> &X, const vector<int> &y)
    {
        if (X.empty() || y.empty())
        {
            root = nullptr;
            return;
        }
        actualMaxDepth = 0;
        root = buildTree(X, y, 0);
    }

    int predict(const vector<double> &sample) const
    {
        if (!root)
            throw std::runtime_error("Tree not trained. Root is null.");
        TreeNode *node = root.get();
        while (!node->isLeaf)
        {
            if (sample[node->featureIndex] <= node->threshold)
                node = node->left.get();
            else
                node = node->right.get();
        }
        return node->prediction;
    }

    double score(const vector<vector<double>> &X, const vector<int> &y) const
    {
        if (!root)
            return 0.0;
        int correct = 0;
        for (size_t i = 0; i < X.size(); ++i)
            if (predict(X[i]) == y[i])
                ++correct;
        return static_cast<double>(correct) / X.size();
    }

    void evaluateDetailed(const vector<vector<double>> &X, const vector<int> &y, const string &outputFilename = "") const
    {
        if (!root || X.empty() || y.empty())
        {
            std::ostringstream oss;
            oss << "Confusion Matrix:\nTree not trained or data empty — evaluation skipped.\n";

            std::cout << oss.str();
            if (!outputFilename.empty())
            {
                std::ofstream out(outputFilename);
                out << oss.str();
            }
            return;
        }
        int TP = 0, TN = 0, FP = 0, FN = 0;
        for (size_t i = 0; i < X.size(); ++i)
        {
            int pred = predict(X[i]);
            if (pred == 1 && y[i] == 1)
                TP++;
            else if (pred == 0 && y[i] == 0)
                TN++;
            else if (pred == 1 && y[i] == 0)
                FP++;
            else if (pred == 0 && y[i] == 1)
                FN++;
        }

        double precision = TP + FP > 0 ? static_cast<double>(TP) / (TP + FP) : 0.0;
        double recall = TP + FN > 0 ? static_cast<double>(TP) / (TP + FN) : 0.0;
        double f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0.0;
        double accuracy = static_cast<double>(TP + TN) / (TP + TN + FP + FN);

        std::ostringstream oss;
        oss << "Confusion Matrix:\n";
        oss << "TP: " << TP << "  FP: " << FP << "\n";
        oss << "FN: " << FN << "  TN: " << TN << "\n";
        oss << "Accuracy: " << accuracy * 100 << "%\n";
        oss << "Precision: " << precision * 100 << "%\n";
        oss << "Recall: " << recall * 100 << "%\n";
        oss << "F1 Score: " << f1 * 100 << "%\n";

        std::cout << oss.str();

        if (!outputFilename.empty())
        {
            std::ofstream out(outputFilename);
            out << oss.str();
        }
    }

    void saveTreeToFile(const string &filename) const
    {
        std::ofstream out(filename);
        printTreeToFileHelper(root.get(), 0, out, "root");
    }

private:
    std::unique_ptr<TreeNode> root;
    int maxDepth;
    int actualMaxDepth;
    vector<string> featureNames;

    std::unique_ptr<TreeNode> buildTree(const vector<vector<double>> &X, const vector<int> &y, int depth)
    {
        actualMaxDepth = std::max(actualMaxDepth, depth);
        auto node = std::make_unique<TreeNode>();

        int ones = std::count(y.begin(), y.end(), 1);
        int zeros = y.size() - ones;
        int majority = (ones >= zeros) ? 1 : 0;

        if (depth >= maxDepth || ones == 0 || zeros == 0)
        {
            node->isLeaf = true;
            node->prediction = majority;
            return node;
        }

        int bestFeature = -1;
        double bestThreshold = 0.0;
        double bestGini = std::numeric_limits<double>::max();
        vector<int> leftIdx, rightIdx;

        for (size_t f = 0; f < X[0].size(); ++f)
        {
            vector<double> values;
            for (const auto &row : X)
                values.push_back(row[f]);
            std::sort(values.begin(), values.end());
            for (size_t i = 1; i < values.size(); ++i)
            {
                double threshold = (values[i - 1] + values[i]) / 2;
                vector<int> left, right;
                for (size_t j = 0; j < X.size(); ++j)
                {
                    if (X[j][f] <= threshold)
                        left.push_back(j);
                    else
                        right.push_back(j);
                }
                if (left.empty() || right.empty())
                    continue;
                double gini = computeGini(y, left, right);
                if (gini < bestGini)
                {
                    bestGini = gini;
                    bestFeature = f;
                    bestThreshold = threshold;
                    leftIdx = left;
                    rightIdx = right;
                }
            }
        }

        if (bestFeature == -1)
        {
            node->isLeaf = true;
            node->prediction = majority;
            return node;
        }

        node->featureIndex = bestFeature;
        node->threshold = bestThreshold;
        node->left = buildTree(extractRows(X, leftIdx), extractLabels(y, leftIdx), depth + 1);
        node->right = buildTree(extractRows(X, rightIdx), extractLabels(y, rightIdx), depth + 1);
        return node;
    }

    double computeGini(const vector<int> &y, const vector<int> &left, const vector<int> &right)
    {
        auto gini = [](const vector<int> &subset, const vector<int> &y)
        {
            if (subset.empty())
                return 0.0;
            int count1 = 0;
            for (int i : subset)
                if (y[i] == 1)
                    ++count1;
            double p = static_cast<double>(count1) / subset.size();
            return 1.0 - (p * p + (1 - p) * (1 - p));
        };
        double gL = gini(left, y);
        double gR = gini(right, y);
        double total = left.size() + right.size();
        return (left.size() / total) * gL + (right.size() / total) * gR;
    }

    vector<vector<double>> extractRows(const vector<vector<double>> &X, const vector<int> &idx)
    {
        vector<vector<double>> out;
        for (int i : idx)
            out.push_back(X[i]);
        return out;
    }

    vector<int> extractLabels(const vector<int> &y, const vector<int> &idx)
    {
        vector<int> out;
        for (int i : idx)
            out.push_back(y[i]);
        return out;
    }

    void printTreeToFileHelper(TreeNode *node, int indent, std::ostream &out, const string &label, const string &side = "") const
    {
        if (!node)
            return;
        string padding(indent * 2, ' ');
        out << padding << label;
        if (!side.empty())
            out << " (" << side << ")";
        out << ": ";
        if (node->isLeaf)
        {
            out << "Predict: " << node->prediction << "\n";
        }
        else
        {
            out << "[X" << node->featureIndex << " (" << featureNames[node->featureIndex] << ") <= " << node->threshold << "]\n";
            printTreeToFileHelper(node->left.get(), indent + 1, out, "if", "left");
            printTreeToFileHelper(node->right.get(), indent + 1, out, "else", "right");
        }
    }
};

void loadData(const string &filename, vector<vector<double>> &features, vector<int> &labels)
{
    std::ifstream file(filename);
    string line;
    std::getline(file, line);
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        string item;
        vector<double> row;
        for (int i = 0; i < 6; ++i)
        {
            std::getline(ss, item, ',');
            row.push_back(std::stod(item));
        }
        std::getline(ss, item, ',');
        row.push_back(item == "Returning_Visitor" ? 1.0 : 0.0);
        std::getline(ss, item, ',');
        row.push_back(std::stoi(item));
        std::getline(ss, item, ',');
        labels.push_back(std::stoi(item));
        features.push_back(row);
    }
}

void appendDataToFile(const string &srcFile, const string &cumulativeFile)
{
    std::ifstream src(srcFile);
    std::ofstream dst(cumulativeFile, std::ios::app);
    string line;
    std::getline(src, line);
    while (std::getline(src, line))
        dst << line << "\n";
}

void copyFile(const string &from, const string &to)
{
    std::ifstream src(from, std::ios::binary);
    std::ofstream dst(to, std::ios::binary);
    dst << src.rdbuf();
}

void evaluateTreeOnData(const DecisionTreeClassifier &clf, const string &dataFile, const string &reportFile = "")
{
    vector<vector<double>> X;
    vector<int> y;
    loadData(dataFile, X, y);
    clf.evaluateDetailed(X, y, reportFile);
}

int main()
{
    vector<string> datasets = {
        "shoppers_train.csv",
        "shoppers_actual.csv",
        "shoppers_multi.csv"};

    vector<string> featureNames = {
        "Administrative", "Product", "Information",
        "BounceRate", "ExitRate", "PageValue",
        "VisitorType", "Weekend"};

    std::ofstream summary("depth_summary.csv");
    summary << "Depth,Round,Accuracy,Precision,Recall,F1\n";

    for (int depth = 1; depth <= 15; ++depth)
    {
        std::cout << "\n==== DEPTH: " << depth << " ====\n"
                  << std::endl;

        string folder = "depth_" + std::to_string(depth);
        system(("mkdir -p " + folder).c_str());

        string cumulativeFile = folder + "/shopper_all_data.csv";
        std::ofstream init(cumulativeFile);
        init << "Administrative,Product,Information,BounceRate,ExitRate,PageValue,VisitorType,Weekend,purchase\n";

        DecisionTreeClassifier masterTree(depth, featureNames);

        int reTest = 0;

        for (size_t k = 0; k < datasets.size(); ++k)
        {
            std::cout << "\n== Round " << k + 1 << " ==\n";

            string confusionFile = folder + "/Tree_R" + std::to_string(k + 1) + "_Metrics.txt";
            vector<vector<double>> X_eval;
            vector<int> y_eval;
            loadData(datasets[k], X_eval, y_eval);

            masterTree.evaluateDetailed(X_eval, y_eval, confusionFile); // Print and save

            // Re-parse metrics from file for CSV summary
            std::ifstream confIn(confusionFile);
            string line;
            double acc = 0, prec = 0, rec = 0, f1 = 0;
            while (std::getline(confIn, line))
            {
                if (line.find("Accuracy:") != std::string::npos)
                    acc = std::stod(line.substr(line.find(":") + 1));
                else if (line.find("Precision:") != std::string::npos)
                    prec = std::stod(line.substr(line.find(":") + 1));
                else if (line.find("Recall:") != std::string::npos)
                    rec = std::stod(line.substr(line.find(":") + 1));
                else if (line.find("F1 Score:") != std::string::npos)
                    f1 = std::stod(line.substr(line.find(":") + 1));
            }
            summary << depth << "," << k + 1 << "," << acc << "," << prec << "," << rec << "," << f1 << "\n";
            if (reTest != 1) // To keep it from double training on itself
            {
                appendDataToFile(datasets[k], cumulativeFile);
            }

            vector<vector<double>> X_cumulative;
            vector<int> y_cumulative;
            loadData(cumulativeFile, X_cumulative, y_cumulative);
            masterTree.fit(X_cumulative, y_cumulative);

            masterTree.saveTreeToFile(folder + "/Tree_Master.txt");
            string treeSnapshot = folder + "/Tree_R" + std::to_string(k + 1) + ".txt";
            copyFile(folder + "/Tree_Master.txt", treeSnapshot);

            if (reTest == 0)
            {
                reTest = 1;
                k = -1;
            }
        }
    }

    return 0;
}
