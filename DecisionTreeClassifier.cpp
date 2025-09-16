// Classifier

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
#include <filesystem>

using std::string;
using std::vector;

struct EvalResult
{
    int TP{0}, TN{0}, FP{0}, FN{0};
    double precision{0.0}, recall{0.0}, f1{0.0}, acc{0.0};
};

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

    void fit(const vector<vector<double>> &x, const vector<int> &y)
    {
        if (x.empty() || y.empty())
        {
            root = nullptr;
            return;
        }
        actualMaxDepth = 0;
        root = buildTree(x, y, 0);
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

    double score(const vector<vector<double>> &x, const vector<int> &y) const
    {
        if (!root)
            return 0.0;
        int correct = 0;
        for (size_t i = 0; i < x.size(); ++i)
            if (predict(x[i]) == y[i])
                ++correct;
        return static_cast<double>(correct) / x.size();
    }

    void evaluateDetailed(const vector<vector<double>> &metrics, const vector<int> &labels, EvalResult &r) const
    {
        if (!root || metrics.empty() || labels.empty())
        {
            std::cout << "Confusion Matrix:\nTree not trained or data empty — evaluation skipped.\n";
            return;
        }
        r = {};
        for (size_t i = 0; i < metrics.size(); ++i)
        {
            int pred = predict(metrics[i]);
            if (pred == 1 && labels[i] == 1)
                r.TP++;
            else if (pred == 0 && labels[i] == 0)
                r.TN++;
            else if (pred == 1 && labels[i] == 0)
                r.FP++;
            else if (pred == 0 && labels[i] == 1)
                r.FN++;
        }

        const int denomP = r.TP + r.FP;
        const int denomR = r.TP + r.FN;
        const int total = r.TP + r.TN + r.FP + r.FN;

        r.precision = denomP ? static_cast<double>(r.TP) / denomP : 0.0;
        r.recall = denomR ? static_cast<double>(r.TP) / denomR : 0.0;
        const double prSum = r.precision + r.recall;
        r.f1 = prSum ? 2.0 * (r.precision * r.recall) / prSum : 0.0;
        r.acc = total ? static_cast<double>(r.TP + r.TN) / total : 0.0;

        std::cout << "Confusion Matrix:\n"
                  << "TP: " << r.TP << "  FP: " << r.FP << "\n"
                  << "FN: " << r.FN << "  TN: " << r.TN << "\n"
                  << "Accuracy: " << (r.acc * 100.0) << "%\n"
                  << "Precision: " << (r.precision * 100.0) << "%\n"
                  << "Recall: " << (r.recall * 100.0) << "%\n"
                  << "F1 Score: " << (r.f1 * 100.0) << "%\n";
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

    std::unique_ptr<TreeNode> buildTree(const vector<vector<double>> &x, const vector<int> &y, int depth)
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

        for (size_t f = 0; f < x[0].size(); ++f)
        {
            vector<double> values;
            for (const auto &row : x)
                values.push_back(row[f]);
            std::sort(values.begin(), values.end());
            for (size_t i = 1; i < values.size(); ++i)
            {
                double threshold = (values[i - 1] + values[i]) / 2;
                vector<int> left, right;
                for (size_t j = 0; j < x.size(); ++j)
                {
                    if (x[j][f] <= threshold)
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
        node->left = buildTree(extractRows(x, leftIdx), extractLabels(y, leftIdx), depth + 1);
        node->right = buildTree(extractRows(x, rightIdx), extractLabels(y, rightIdx), depth + 1);
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

    vector<vector<double>> extractRows(const vector<vector<double>> &x, const vector<int> &idx)
    {
        vector<vector<double>> out;
        for (int i : idx)
            out.push_back(x[i]);
        return out;
    }

    vector<int> extractLabels(const vector<int> &y, const vector<int> &idx)
    {
        vector<int> out;
        for (int i : idx)
            out.push_back(y[i]);
        return out;
    }

    void printTreeToFileHelper(TreeNode *node, int indent, std::ostream &out, const string &labels, const string &side = "") const
    {
        if (!node)
            return;
        string padding(indent * 2, ' ');
        out << padding << labels;
        if (!side.empty())
            out << " (" << side << ")";
        out << ": ";
        if (node->isLeaf)
        {
            out << "Predict: " << node->prediction << "\n";
        }
        else
        {
            out << "[x" << node->featureIndex << " (" << featureNames[node->featureIndex] << ") <= " << node->threshold << "]\n";
            printTreeToFileHelper(node->left.get(), indent + 1, out, "if", "left");
            printTreeToFileHelper(node->right.get(), indent + 1, out, "else", "right");
        }
    }
};

void loadData(const string &filename, vector<vector<double>> &features, vector<int> &labels)
{
    std::ifstream file(filename);
    string line;
    //  Reads first line and ignores it. Deals with Header
    std::getline(file, line);
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        string item;
        vector<double> row;
        // First six values are doubles
        for (int i = 0; i < 6; ++i)
        {
            std::getline(ss, item, ',');
            row.push_back(std::stod(item));
        }
        // 7th value is a String. Convert to Double
        std::getline(ss, item, ',');
        row.push_back(item == "Returning_Visitor" ? 1.0 : 0.0);
        // Next items is integer. Converts to Double automatically
        std::getline(ss, item, ',');
        row.push_back(std::stoi(item));
        // Whether or not a purchase was made. Goes into labels.
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

void init(vector<string> &datasets, vector<string> &featureNames, string &output)
{
    // Input files
    datasets = {
        "Data_Input/shoppers_train.csv",
        "Data_Input/shoppers_actual.csv"};
    featureNames = {
        "Administrative", "Product", "Information",
        "BounceRate", "ExitRate", "PageValue",
        "VisitorType", "Weekend"};
    // Output file
    output = "Data_Output/";
    std::filesystem::create_directories(output);
    return;
}

int main()
{
    vector<string> datasets;
    vector<string> featureNames;
    string output;
    EvalResult evalResult;
    // Sets up names for datasets, feature names, etc
    init(datasets, featureNames, output);

    std::ofstream summary("depth_summary.csv");
    summary << "Depth,Round,Accuracy,Precision,Recall,F1\n";

    for (int depth = 1; depth <= 15; ++depth)
    {
        std::cout << "\nDEPTH: " << depth << "\n\n";

        string folder = output + "depth_" + std::to_string(depth);
        std::filesystem::create_directories(folder);

        DecisionTreeClassifier masterTree(depth, featureNames);

        for (size_t k = 0; k < datasets.size(); ++k)
        {
            string metricsFile = folder + "/Tree_R" + std::to_string(k + 1) + "_Metrics.txt";
            vector<vector<double>> metrics;
            vector<int> labels;
            loadData(datasets[k], metrics, labels);

            if (k == 0)
            {
                masterTree.fit(metrics, labels);
            }
            masterTree.evaluateDetailed(metrics, labels, evalResult); // Print

            // Re-parse metrics from file for CSV summary
            std::ifstream confIn(metricsFile);
            string line;
            double acc = 0, prec = 0, rec = 0, f1 = 0;
            summary << depth << "," << k + 1 << "," << evalResult.acc << ",";
            summary << evalResult.precision << "," << evalResult.recall << "," << evalResult.f1 << "\n";

            masterTree.saveTreeToFile(folder + "/Tree_Master.txt");
            string treeSnapshot = folder + "/Tree_R" + std::to_string(k + 1) + ".txt";
            copyFile(folder + "/Tree_Master.txt", treeSnapshot);
        }
    }

    return 0;
}
