#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>

using namespace cv;
using namespace std;

// ---------------------------
// Global State and Data Structures
// ---------------------------
enum State { SET_P0, SET_P1, WAIT_STROKE, DRAW_STROKE };
State currentState = SET_P0;

Point2d p0, p1;                    // Endpoints for the line segment
vector<Point2d> currentStroke;     // Freehand stroke points
vector<Point2d> sampledLine;       // Uniformly sampled points from p0 to p1
vector<double> currentzDistances;         // Distances for deformation
vector<vector<double>>allzDistances;
vector<pair<Point2d, Point2d>> lineSegments;  // Stores all drawn line segments
vector<Point2d> intersections;               // Stores intersection points
vector<vector<Point2d>> sampledLines;
pair<int,int> intersectionPair;
// ---------------------------
// Utility Functions
// ---------------------------

// ARAP Deformation for a 1D curve (chain) in 3D using Eigen.
// X0: initial 3D curve as an (n x 3) matrix (each row is a point)
// base: the reference (rest) shape (n x 3) – typically the same as X0
// constraints: indices of vertices that are fixed (e.g., {0, intersectionIndex, n-1})
// targetPositions: a (c x 3) matrix with the target positions for the fixed vertices
// maxIter: maximum iterations (not fully iterated in this simple version)
// tol: convergence tolerance (not used in this one-shot solve version)
Eigen::MatrixXd deformCurveARAP(const Eigen::MatrixXd &X0,
                                const Eigen::MatrixXd &base,
                                const std::vector<int> &constraints,
                                const Eigen::MatrixXd &targetPositions,
                                int maxIter = 10,
                                double tol = 1e-4) {
    int n = X0.rows();
    Eigen::MatrixXd X = X0; // Current deformed positions

    // Build the Laplacian matrix L for a chain (n x n)
    Eigen::SparseMatrix<double> L(n, n);
    std::vector<Eigen::Triplet<double>> triplets;
    if(n > 0) {
        triplets.push_back(Eigen::Triplet<double>(0, 0, 1));
        if(n > 1)
            triplets.push_back(Eigen::Triplet<double>(0, 1, -1));
    }
    for(int i = 1; i < n - 1; i++){
        triplets.push_back(Eigen::Triplet<double>(i, i-1, -1));
        triplets.push_back(Eigen::Triplet<double>(i, i, 2));
        triplets.push_back(Eigen::Triplet<double>(i, i+1, -1));
    }
    if(n > 1) {
        int i = n - 1;
        triplets.push_back(Eigen::Triplet<double>(i, i-1, -1));
        triplets.push_back(Eigen::Triplet<double>(i, i, 1));
    }
    L.setFromTriplets(triplets.begin(), triplets.end());

    // Build the constraint matrix C (c x n) where c = constraints.size()
    int c = constraints.size();
    Eigen::SparseMatrix<double> C(c, n);
    std::vector<Eigen::Triplet<double>> tripC;
    for (int i = 0; i < c; i++){
        int idx = constraints[i];
        tripC.push_back(Eigen::Triplet<double>(i, idx, 1));
    }
    C.setFromTriplets(tripC.begin(), tripC.end());

    // Weight for constraints
    double lambda = 1e6;
    int m = n + c; // Total rows in augmented system

    // Build the augmented matrix A = [L; sqrt(lambda)*C]
    Eigen::SparseMatrix<double> A(m, n);
    std::vector<Eigen::Triplet<double>> tripA;
    for (int k = 0; k < L.outerSize(); k++){
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it){
            tripA.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
        }
    }
    for (int k = 0; k < C.outerSize(); k++){
        for (Eigen::SparseMatrix<double>::InnerIterator it(C, k); it; ++it){
            tripA.push_back(Eigen::Triplet<double>(L.rows() + it.row(), it.col(), std::sqrt(lambda) * it.value()));
        }
    }
    A.setFromTriplets(tripA.begin(), tripA.end());

    // Construct the right-hand side b.
    // Here, we use the Laplacian applied to the base as a simple divergence vector.
    // In a full ARAP, you would compute per-edge rotations and then build b accordingly.
    Eigen::MatrixXd b = L * base;
    // Augment b with the constraints.
    Eigen::MatrixXd b_aug(m, 3);
    b_aug.topRows(n) = b;
    b_aug.bottomRows(c) = std::sqrt(lambda) * targetPositions;

    // Solve the normal equations: (A^T A) X = A^T b_aug.
    Eigen::SparseMatrix<double> AtA = A.transpose() * A;
    Eigen::MatrixXd Atb = A.transpose() * b_aug;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(AtA);
    if(solver.info() != Eigen::Success) {
        std::cerr << "Decomposition failed in ARAP." << std::endl;
        return X;
    }
    X = solver.solve(Atb);
    if(solver.info() != Eigen::Success) {
        std::cerr << "Solving failed in ARAP." << std::endl;
        return X;
    }
    return X;
}

Eigen::MatrixXd vectorToMat(const vector<Point3d>& pts) {
    int n = pts.size();
    Eigen::MatrixXd M(n, 3);
    for (int i = 0; i < n; i++) {
        M(i, 0) = pts[i].x;
        M(i, 1) = pts[i].y;
        M(i, 2) = pts[i].z;
    }
    return M;
}

vector<Point3d> matToVector(const Eigen::MatrixXd &M) {
    int n = M.rows();
    vector<Point3d> pts;
    for (int i = 0; i < n; i++) {
        pts.push_back(Point3d(M(i,0), M(i,1), M(i,2)));
    }
    return pts;
}


void exportToOBJ(const vector<Point3d>& curve, const string& filename) {
    ofstream file(filename);

    int vertexOffset = 1;
    // for (const auto& curve : curves)
    {
        // Write vertices
        for (const auto& p : curve) {
            file << "v " << p.x << " " << p.y << " " << p.z << "\n";
        }

        // Write line (connectivity)

        for (size_t i = 1; i < curve.size(); ++i) {
            file << "l " <<i<<" "<< vertexOffset + i << "\n";
        }

    }
}

// Uniformly sample a line with 'n' points
vector<Point2d> sampleLine2D(const Point2d& start, const Point2d& end, int n) {
    vector<Point2d> pts;
    for (int i = 0; i < n; i++) {
        double t = double(i) / (n - 1);
        pts.push_back(Point2d(start.x * (1 - t) + end.x * t,
                              start.y * (1 - t) + end.y * t));
    }
    return pts;
}

// Convert vector<Point2d> to vector<Point> (for OpenCV drawing)
vector<Point> convertToIntPoints(const vector<Point2d>& pts) {
    vector<Point> ret;
    for (const auto& p : pts)
        ret.push_back(Point(cvRound(p.x), cvRound(p.y)));
    return ret;
}

// Compute Euclidean distance between two points
double distance2D(const Point2d& a, const Point2d& b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

// ---------------------------
// Line Intersection Helper
// ---------------------------

// Check if two line segments (A-B and C-D) intersect
bool lineSegmentIntersection(Point2d A, Point2d B, Point2d C, Point2d D, Point2d& intersection) {
    // Solve for intersection using determinant approach
    double denom = (A.x - B.x) * (C.y - D.y) - (A.y - B.y) * (C.x - D.x);
    if (denom == 0) return false; // Parallel lines

    double t = ((A.x - C.x) * (C.y - D.y) - (A.y - C.y) * (C.x - D.x)) / denom;
    double u = ((A.x - C.x) * (A.y - B.y) - (A.y - C.y) * (A.x - B.x)) / denom;

    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
        intersection.x = A.x + t * (B.x - A.x);
        intersection.y = A.y + t * (B.y - A.y);
        return true;
    }
    return false;
}

// ---------------------------
// Redraw function – draws current state
// ---------------------------
void redraw() {
    Mat display(480, 640, CV_8UC3, Scalar(255,255,255));

    // Draw existing line segments
    for (const auto& segment : lineSegments) {
        line(display, segment.first, segment.second, Scalar(255,0,0), 2);
    }

    // Draw current line segment if endpoints are set
    if (currentState == SET_P1 || currentState == WAIT_STROKE || currentState == DRAW_STROKE) {
        line(display, p0, p1, Scalar(0,255,0), 2);
        circle(display, p0, 3, Scalar(0,255,0), -1);
        circle(display, p1, 3, Scalar(0,255,0), -1);
    }

    // Draw current stroke
    if (!currentStroke.empty()) {
        polylines(display, convertToIntPoints(currentStroke), false, Scalar(0,255,0), 1, LINE_AA);
    }

    // Draw red lines connecting sampled points and stroke points
    if (!sampledLine.empty() && sampledLine.size() == currentStroke.size()) {
        vector<Point> samplePts = convertToIntPoints(sampledLine);
        vector<Point> strokePts = convertToIntPoints(currentStroke);
        for (size_t i = 0; i < samplePts.size(); i++) {
            line(display, samplePts[i], strokePts[i], Scalar(0,0,255), 1);
        }
    }

    // Draw intersection points in RED
    for (const auto& pt : intersections) {
        circle(display, pt, 5, Scalar(0,0,255), -1);
    }

    imshow("Curve Deformation", display);
}

// ---------------------------
// Mouse Callback – Implements the state machine
// ---------------------------
void mouseCallback(int event, int x, int y, int flags, void*) {
    Point2d pos(x, y);

    if (currentState == SET_P0 && event == EVENT_LBUTTONDOWN) {
        p0 = pos;
        cout << "p0 set to " << p0 << endl;
        currentState = SET_P1;
    }
    else if (currentState == SET_P1 && event == EVENT_LBUTTONDOWN) {
        p1 = pos;
        cout << "p1 set to " << p1 << endl;
        currentState = WAIT_STROKE;
        lineSegments.push_back({p0, p1}); // Store the line segment
        cout << "Press 'S' to start drawing stroke." << endl;
    }
    else if (currentState == DRAW_STROKE) {
        if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {
            currentStroke.push_back(pos);
        }
    }

    redraw();
}

// ---------------------------
// Keyboard Callback
// ---------------------------
void keyCallback(int key) {
    if (key == 's' || key == 'S') {
        if (currentState == WAIT_STROKE) {
            cout << "Start drawing the stroke..." << endl;
            currentStroke.clear();
            currentState = DRAW_STROKE;
        }
    }
    else if (key == 13) { // Enter key to finish stroke
        if (currentState == DRAW_STROKE) {
            if (currentStroke.size() < 2) {
                cout << "Not enough points drawn. Resetting." << endl;
                currentState = SET_P0;
                return;
            }

            // Sample the segment with the same number of points as the stroke
            sampledLine = sampleLine2D(p0, p1, currentStroke.size());
            sampledLines.push_back(sampledLine);

            // Compute distances
            currentzDistances.clear();
            for (size_t i = 0; i < sampledLine.size(); i++) {
                currentzDistances.push_back(distance2D(sampledLine[i], currentStroke[i]));
            }
            allzDistances.push_back(currentzDistances);
            cout << "Stroke finished. Press 'I' to check intersections." << endl;
            currentState = SET_P0;
        }
    }
    else if (key == 'i' || key == 'I') { // Check intersection
        if (sampledLines.size() < 2) {
            cout << "Not enough sampled lines to check intersections." << endl;
            return;
        }

        intersections.clear();

        // Check for intersections between the first and second sampled line segments
        const auto& line1 = sampledLines[0];
        const auto& line2 = sampledLines[1];

        for (size_t i = 1; i < line1.size(); i++) {
            Point2d segA_start = line1[i - 1];
            Point2d segA_end = line1[i];

            for (size_t j = 1; j < line2.size(); j++) {
                Point2d segB_start = line2[j - 1];
                Point2d segB_end = line2[j];

                Point2d intersection;
                if (lineSegmentIntersection(segA_start, segA_end, segB_start, segB_end, intersection)) {
                    intersections.push_back(intersection);
                    cout << "Intersection at: " << intersection << endl;
                    intersectionPair.first = i;
                    intersectionPair.second = j;
                }
            }
        }

        cout << "Total intersections found: " << intersections.size() << endl;
        redraw();
    }

    else if (key == 'o' || key == 'O') { // Export 3D curves as OBJ
        if (sampledLines.size() < 2) {
            cout << "Not enough sampled lines to create 3D curves." << endl;
            return;
        }

        // Create 3D curve by using z-coordinates from the sampled lines
        vector<Point3d> firstcurve; vector<Point3d> firstcurvewithz;
        vector<Point3d> secondcurve; vector<Point3d> secondcurvewithz;
        vector<Point3d> testcurve;
        const auto& line1 = sampledLines[0];
        const auto& line2 = sampledLines[1];

        for (size_t i = 0; i < line1.size(); ++i) {
            double z = 0;
            if(i<allzDistances[0].size())
                z = allzDistances[0][i];
            // if(i==intersectionPair.second)
            //     z = allzDistances[0][intersectionPair.first];
            firstcurve.push_back(Point3d(line1[i].x, line1[i].y, 0.0));  // Add the 3D point
            firstcurvewithz.push_back(Point3d(line1[i].x, line1[i].y, z));  // Add the 3D point
        }

        for (size_t i = 0; i < line2.size(); ++i) {
            double z = 0;
            if(i<allzDistances[1].size())
                z = allzDistances[1][i];
            if(i==intersectionPair.second)
                testcurve.push_back(Point3d(line2[i].x, line2[i].y, allzDistances[0][intersectionPair.first]));  // Add the 3D point
            else
                testcurve.push_back(Point3d(line2[i].x, line2[i].y, z));  // Add the 3D point
            //     z = allzDistances[0][intersectionPair.first];

            secondcurve.push_back(Point3d(line2[i].x, line2[i].y, 0.0));  // Add the 3D point
            secondcurvewithz.push_back(Point3d(line2[i].x, line2[i].y, z));  // Add the 3D point
        }

        std::vector<int> constraints;
        constraints.push_back(0);
        constraints.push_back(secondcurvewithz.size()-1);
        constraints.push_back(intersectionPair.second);

        vector<Point3d>target;
        target.push_back(secondcurvewithz[0]);
        target.push_back(secondcurvewithz[secondcurvewithz.size()-1]);
        target.push_back(Point3d(line2[intersectionPair.second].x, line2[intersectionPair.second].y,allzDistances[0][intersectionPair.first]));


        vector<Point3d> deformed_3DCurve = matToVector(deformCurveARAP(vectorToMat(secondcurvewithz), vectorToMat(secondcurvewithz), constraints, vectorToMat(target)));

        // Export to OBJ file
        exportToOBJ(firstcurve, "Firstcurve.obj");
        exportToOBJ(firstcurvewithz, "Firstcurvewithz.obj");
        exportToOBJ(secondcurve, "secondcurve.obj");
        exportToOBJ(secondcurvewithz, "secondcurvewithz.obj");
        exportToOBJ(testcurve, "testcurve.obj");
        exportToOBJ(deformed_3DCurve, "deformedSecosndcurve.obj");
        cout << "obj generated" << endl;

        redraw();
    }

}

// ---------------------------
// Main Function
// ---------------------------
int main() {
    namedWindow("Curve Deformation", WINDOW_AUTOSIZE);
    setMouseCallback("Curve Deformation", mouseCallback);

    Mat display(480, 640, CV_8UC3, Scalar(255,255,255));
    imshow("Curve Deformation", display);

    cout << "Instructions:\n"
         << "1. Click to set the first endpoint (p0).\n"
         << "2. Click to set the second endpoint (p1).\n"
         << "3. Press 'S' to start drawing the stroke.\n"
         << "4. Drag the mouse while holding left-click to draw.\n"
         << "5. Press 'Enter' to finish the stroke (red lines appear).\n"
         << "Press ESC to exit.\n";

    while (true) {
        int key = waitKey(1);
        if (key == 27) break; // ESC to exit
        keyCallback(key);
    }

    return 0;
}
