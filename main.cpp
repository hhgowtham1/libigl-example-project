#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <Eigen/Sparse>

using namespace cv;
using namespace std;

// Data structures
struct CurveData {
    Point2d p0, p1;
    vector<Point2d> sampledLine;
    vector<Point2d> strokePoints;
    vector<double> zDistances;
    vector<Point3d> deformedCurve3D;
};

struct Intersection {
    int curveIndex;
    int pointIndex;
    Point2d position;
};

// Global state
vector<CurveData> curves;
vector<Intersection> currentIntersections;
Point2d p0, p1;
vector<Point2d> currentStroke;
enum State { SET_P0, SET_P1, DRAW_STROKE };
State currentState = SET_P0;

// Utility functions
Point2d cv2gl(const Point2d& p) { return Point2d(p.x, -p.y + 480); }
Point2d gl2cv(const Point2d& p) { return Point2d(p.x, -p.y + 480); }

vector<Point2d> sampleLine(const Point2d& start, const Point2d& end, int n) {
    vector<Point2d> points;
    for(int i = 0; i < n; ++i) {
        double t = static_cast<double>(i)/(n-1);
        points.emplace_back(start.x*(1-t) + end.x*t,
                           start.y*(1-t) + end.y*t);
    }
    return points;
}

bool segmentIntersection(const Point2d& a1, const Point2d& a2,
                        const Point2d& b1, const Point2d& b2,
                        Point2d& intersection) {
    Point2d da = a2 - a1;
    Point2d db = b2 - b1;
    Point2d ab = b1 - a1;

    double cross = da.x*db.y - da.y*db.x;
    if(abs(cross) < 1e-7) return false;

    double t = (ab.x*db.y - ab.y*db.x)/cross;
    double u = (ab.x*da.y - ab.y*da.x)/cross;
    
    if(t >= 0 && t <= 1 && u >= 0 && u <= 1) {
        intersection = a1 + t*da;
        return true;
    }
    return false;
}

// ARAP Deformation
vector<Point3d> deformARAP(const vector<Point3d>& curve,
                          const vector<pair<int, Point3d>>& constraints) {
    const int n = curve.size();
    typedef Eigen::Triplet<double> T;
    
    // Build Laplace matrix
    Eigen::SparseMatrix<double> L(n, n);
    vector<T> coefficients;
    for(int i = 1; i < n-1; ++i) {
        coefficients.emplace_back(i, i-1, -1);
        coefficients.emplace_back(i, i, 2);
        coefficients.emplace_back(i, i+1, -1);
    }
    L.setFromTriplets(coefficients.begin(), coefficients.end());
    
    // Build constraint matrix
    Eigen::SparseMatrix<double> C(constraints.size(), n);
    Eigen::VectorXd d(constraints.size());
    for(size_t i = 0; i < constraints.size(); ++i) {
        C.insert(i, constraints[i].first) = 1;
        d[i] = constraints[i].second.z;
    }
    
    // Solve (L^T L + λ C^T C) x = L^T L x0 + λ C^T d
    const double lambda = 1e6;
    Eigen::SparseMatrix<double> A = L.transpose() * L + lambda * C.transpose() * C;
    Eigen::VectorXd b = L.transpose() * L * Eigen::Map<const Eigen::VectorXd>(&curve[0].x, n*3)
                      + lambda * C.transpose() * d;
    
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    Eigen::VectorXd x = solver.solve(b);
    
    // Reconstruct deformed curve
    vector<Point3d> deformed;
    for(int i = 0; i < n; ++i) {
        deformed.emplace_back(x[3*i], x[3*i+1], curve[i].z);
    }
    return deformed;
}
void redraw() {
    Mat display(480, 640, CV_8UC3, Scalar(255,255,255));
    
    // Draw existing curves
    for(const auto& curve : curves) {
        line(display, cv2gl(curve.p0), cv2gl(curve.p1), Scalar(255,0,0), 2);
    }
    
    // Draw current line segment
    if(currentState == SET_P1 || currentState == DRAW_STROKE) {
        line(display, cv2gl(p0), cv2gl(p1), Scalar(0,255,0), 2);
    }
    
    imshow("Curve Deformation", display);
}
// OBJ Export
void exportToOBJ(const vector<CurveData>& curves, const string& filename) {
    ofstream file(filename);
    int vertexOffset = 1;
    
    for(const auto& curve : curves) {
        // Write vertices
        for(const auto& p : curve.deformedCurve3D) {
            file << "v " << p.x << " " << p.y << " " << p.z << "\n";
        }
        
        // Write line
        file << "l";
        for(size_t i = 0; i < curve.deformedCurve3D.size(); ++i) {
            file << " " << vertexOffset + i;
        }
        file << "\n";
        
        vertexOffset += curve.deformedCurve3D.size();
    }
}

// Mouse callback
void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    Point2d p(x, y);
    cout << "Mouse event: " << event << " at (" << x << "," << y << ")" << endl;
        cout << "Current state: " << currentState << endl;
    switch(currentState) {
        case SET_P0:
            if(event == EVENT_LBUTTONDOWN) {
                p0 = gl2cv(p);
                currentState = SET_P1;
                cout << "Set p0 to " << p0 << endl;
                redraw();
            }
            break;
            
        case SET_P1:
            if(event == EVENT_LBUTTONDOWN) {
                p1 = gl2cv(p);
                currentState = DRAW_STROKE;
                cout << "Set p1 to " << p1 << endl;
                redraw();
            }
            break;
            
        case DRAW_STROKE:
            if(event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {
                currentStroke.push_back(gl2cv(p));
                redraw();
            }
            else if(event == EVENT_LBUTTONUP) {
                // Process new curve
                CurveData newCurve;
                newCurve.p0 = p0;
                newCurve.p1 = p1;
                newCurve.sampledLine = sampleLine(p0, p1, currentStroke.size());
                newCurve.strokePoints = currentStroke;
                
                // Calculate z distances
                for(size_t i = 0; i < newCurve.sampledLine.size(); ++i) {
                    double dx = newCurve.strokePoints[i].x - newCurve.sampledLine[i].x;
                    double dy = newCurve.strokePoints[i].y - newCurve.sampledLine[i].y;
                    newCurve.zDistances.push_back(sqrt(dx*dx + dy*dy));
                }
                
                // Create initial 3D curve (x,y from sampled line, z from distances)
                vector<Point3d> initial3D;
                for(size_t i = 0; i < newCurve.sampledLine.size(); ++i) {
                    initial3D.emplace_back(
                        newCurve.sampledLine[i].x,
                        newCurve.sampledLine[i].y,
                        newCurve.zDistances[i]
                    );
                }
                
                // Check intersections with existing curves
                vector<pair<int, Point3d>> constraints;
                constraints.emplace_back(0, initial3D.front());
                constraints.emplace_back(initial3D.size()-1, initial3D.back());
                
                for(const auto& existing : curves) {
                    Point2d intersect;
                    if(segmentIntersection(existing.p0, existing.p1, p0, p1, intersect)) {
                        // Find closest point on new curve
                        int closestIdx = 0;
                        double minDist = numeric_limits<double>::max();
                        for(size_t i = 0; i < newCurve.sampledLine.size(); ++i) {
                            double d = norm(newCurve.sampledLine[i] - intersect);
                            if(d < minDist) {
                                minDist = d;
                                closestIdx = i;
                            }
                        }
                        
                        // Add constraint using existing curve's z-distance
                        constraints.emplace_back(closestIdx,
                            Point3d(intersect.x, intersect.y, existing.zDistances[closestIdx]));
                    }
                }
                
                // Apply ARAP deformation if needed
                if(constraints.size() > 2) {
                    newCurve.deformedCurve3D = deformARAP(initial3D, constraints);
                } else {
                    newCurve.deformedCurve3D = initial3D;
                }
                
                curves.push_back(newCurve);
                exportToOBJ(curves, "output.obj");
                
                // Reset for next curve
                currentState = SET_P0;
                currentStroke.clear();
                redraw();
            }
            break;
    }
    
    // Redraw
    Mat display(480, 640, CV_8UC3, Scalar(255,255,255));
    
    // Draw existing curves
    for(const auto& curve : curves) {
        // Draw base line
        line(display, cv2gl(curve.p0), cv2gl(curve.p1), Scalar(255,0,0));
        
        // Draw deformed 3D curve projection
        vector<Point2d> projPoints;
        for(const auto& p : curve.deformedCurve3D) {
            projPoints.push_back(cv2gl(Point2d(p.x, p.y)));
        }
        polylines(display, projPoints, false, Scalar(0,255,255));
    }
    
    // Draw current state
    if(currentState == SET_P1) {
        circle(display, Point(x,y), 3, Scalar(0,0,255), -1);
    }
    else if(currentState == DRAW_STROKE) {
        // Draw sampled line
        vector<Point2d> glPoints;
        for(const auto& p : sampleLine(p0, p1, currentStroke.size())) {
            glPoints.push_back(cv2gl(p));
        }
        polylines(display, glPoints, false, Scalar(255,0,0));
        
        // Draw stroke
        vector<Point> cvPoints;
        for(const auto& p : currentStroke) {
            cvPoints.push_back(cv2gl(p));
        }
        polylines(display, cvPoints, false, Scalar(0,255,0), 1, LINE_AA);
        
        // Draw connectors
        if(currentStroke.size() == sampleLine(p0, p1, currentStroke.size()).size()) {
            auto sampled = sampleLine(p0, p1, currentStroke.size());
            for(size_t i = 0; i < sampled.size(); ++i) {
                line(display, cv2gl(sampled[i]), cv2gl(currentStroke[i]),
                    Scalar(0,0,255));
            }
        }
    }
    
    imshow("Curve Deformation", display);
}

int main() {
    namedWindow("Curve Deformation", WINDOW_AUTOSIZE);
        setMouseCallback("Curve Deformation", mouseCallback);
        
        Mat display(480, 640, CV_8UC3, Scalar(255,255,255));
        imshow("Curve Deformation", display);
        
        while(waitKey(1) != 27) {} // ESC to exit
        
        return 0;
}
