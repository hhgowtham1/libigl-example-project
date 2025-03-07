#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include<variant>
#include <algorithm>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Constrained_triangulation_plus_2.h>
#include <CGAL/intersections.h>
#include <igl/arap.h>
#include <Eigen/Core>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

typedef CGAL::Exact_predicates_exact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_2<unsigned, K> Vb;
typedef CGAL::Constrained_triangulation_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Exact_intersections_tag Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds, Itag> CDT;
typedef CGAL::Constrained_triangulation_plus_2<CDT> CDT_plus;
typedef CDT_plus::Point Point;


// Assuming you're using the CGAL::Exact_predicates_inexact_constructions_kernel
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_2 CGAL_Point_2;
typedef Kernel::Segment_2 CGAL_Segment_2;

std::vector<cv::Point> gridSampling(const cv::Mat& region, float radius1=20, int numSamples = 3000)
{
  std::vector<cv::Point> samples;
    int radius=20;
    for (int y = radius; y < region.rows; y += 2*radius)
        for (int x = radius; x < region.cols; x += 2*radius)
            if (region.at<uchar>(y, x) > 0)
                samples.push_back(cv::Point(x, y));
    return samples;
}
cv::Mat displayImage; // Global variable to store the image being displayed
cv::Point firstPoint(-1, -1),secondPoint(-1, -1);
cv::Point savedFirstPoint(-1, -1),savedSecondPoint(-1, -1);
cv::Mat floodRegion;
std::vector<cv::Point> polylinePoints;
bool drawingPolyline = false;
bool isDrawing = false;

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <iostream>

void laplacian_polyline_deform(
    const Eigen::MatrixXd& V,  // Original vertex positions (n x 3)
    const Eigen::MatrixXi& E,  // Edge list (m x 2)
    const Eigen::VectorXi& b,  // Indices of constrained vertices
    const Eigen::MatrixXd& bc, // New positions of constrained vertices (b.size() x 3)
    Eigen::MatrixXd& U         // Output deformed vertex positions
) {
    using namespace Eigen;
    int n = V.rows(), m = E.rows();
    U = V; // Initialize output

    // Step 1: Compute Initial Edge Vectors (to preserve relative motion)
    MatrixXd edge_vectors(m, 3);
    for (int i = 0; i < m; ++i) {
        edge_vectors.row(i) = V.row(E(i, 1)) - V.row(E(i, 0));
    }

    // Step 2: Construct Laplacian Matrix
    SparseMatrix<double> L(n, n);
    std::vector<Triplet<double>> triplets;
    std::vector<double> diag(n, 0.0);

    for (int i = 0; i < m; ++i) {
        int vi = E(i, 0), vj = E(i, 1);
        triplets.emplace_back(vi, vj, -1.0);
        triplets.emplace_back(vj, vi, -1.0);
        diag[vi] += 1.0;
        diag[vj] += 1.0;
    }

    for (int i = 0; i < n; ++i) {
        if (diag[i] > 0) {
            triplets.emplace_back(i, i, diag[i]);
        } else {
            triplets.emplace_back(i, i, 1.0);
        }
    }

    L.setFromTriplets(triplets.begin(), triplets.end());

    // Step 3: Construct RHS with Edge Constraints
    MatrixXd rhs = L * V;

    // Step 4: Apply Constraints (Hard boundary enforcement)
    for (int i = 0; i < b.size(); ++i) {
        int idx = b(i);
        L.coeffRef(idx, idx) = 1.0;
        rhs.row(idx) = bc.row(i);
    }

    // Step 5: Solve the linear system
    SimplicialLDLT<SparseMatrix<double>> solver;
    solver.compute(L);
    if (solver.info() != Success) {
        std::cerr << "Solver failed!" << std::endl;
        return;
    }

    MatrixXd U_prev = U;

    // Step 6: Iteratively Improve Rigidity
    for (int iter = 0; iter < 10; ++iter) {  // 10 iterations to improve rigidity
        // Solve for new positions
        for (int dim = 0; dim < 3; ++dim) {
            U.col(dim) = solver.solve(rhs.col(dim));
        }

        // Compute new edge vectors
        MatrixXd new_edge_vectors(m, 3);
        for (int i = 0; i < m; ++i) {
            new_edge_vectors.row(i) = U.row(E(i, 1)) - U.row(E(i, 0));
        }

        // Compute edge correction
        for (int i = 0; i < m; ++i) {
            Vector3d correction = (edge_vectors.row(i) - new_edge_vectors.row(i)) * 0.5;
            U.row(E(i, 0)) += correction;
            U.row(E(i, 1)) -= correction;
        }

        // Convergence check
        if ((U - U_prev).norm() < 1e-5) break;
        U_prev = U;
    }
}



void arap_polyline_deform1(
    const Eigen::MatrixXd& V,  // Original vertex positions
    const Eigen::MatrixXi& E,  // Edge list
    Eigen::VectorXi b, // Indices of constrained vertices
    const Eigen::MatrixXd& bc, // New positions of constrained vertices
    Eigen::MatrixXd& U         // Output deformed vertex positions
) {
    // Initialize ARAP data
    igl::ARAPData arap_data;
    arap_data.with_dynamics = false;
    arap_data.max_iter = 10;

    // Precompute ARAP system
    igl::arap_precomputation(V, E, V.cols(), b, arap_data);

    // Perform ARAP deformation
    igl::arap_solve(bc, arap_data, U);
}



// Helper function to find intersection between two line segments
bool lineLineIntersection(cv::Point2f a1, cv::Point2f a2, cv::Point2f b1, cv::Point2f b2, cv::Point2f& intersection)
{
    cv::Point2f r = a2 - a1;
    cv::Point2f s = b2 - b1;
    float rxs = r.x * s.y - r.y * s.x;
    float qpxr = (b1.x - a1.x) * r.y - (b1.y - a1.y) * r.x;

    if (std::abs(rxs) < 1e-8 && std::abs(qpxr) < 1e-8)
        return false; // Collinear

    if (std::abs(rxs) < 1e-8)
        return false; // Parallel

    float t = ((b1.x - a1.x) * s.y - (b1.y - a1.y) * s.x) / rxs;
    float u = qpxr / rxs;

    if (0 <= t && t <= 1 && 0 <= u && u <= 1)
    {
        intersection = a1 + t * r;
        return true;
    }

    return false;
}

// Conversion function from cv::Point2f to CGAL_Point_2
CGAL_Point_2 toCGALPoint(const cv::Point2f& cvPoint) {
    return CGAL_Point_2(cvPoint.x, cvPoint.y);
}

bool lineIntersects(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& q1, const cv::Point2f& q2) {
    CGAL_Segment_2 seg1(toCGALPoint(p1), toCGALPoint(p2));
    CGAL_Segment_2 seg2(toCGALPoint(q1), toCGALPoint(q2));
    auto result = CGAL::intersection(seg1, seg2);
    return result.has_value();
}

// Helper function to find intersection between a line and a polyline
cv::Point2f findIntersection(cv::Point2f start, cv::Point2f direction, const std::vector<cv::Point>& polyline)
{
    for (size_t i = 0; i < polyline.size() - 1; ++i)
    {
        cv::Point2f p1 = polyline[i];
        cv::Point2f p2 = polyline[i + 1];
        cv::Point2f intersection;
        if (lineLineIntersection(start, start + direction, p1, p2, intersection))
        {
            return intersection;
        }
    }
    return start; // Return start point if no intersection found
}

struct PolylineData {
    cv::Point2f firstPoint;
    cv::Point2f secondPoint;
    std::vector<cv::Point3f> polyline3D;
    std::vector<cv::Point2f> samples;
    std::vector<double> distance3D;
};

std::vector<PolylineData> savedPolylines;

cv::Point2f toCVPoint(const CGAL_Point_2& cgalPoint) {
    return cv::Point2f(CGAL::to_double(cgalPoint.x()), CGAL::to_double(cgalPoint.y()));
}

bool findIntersection(const cv::Point2f& p1, const cv::Point2f& p2, 
                      const cv::Point2f& q1, const cv::Point2f& q2, 
                      cv::Point2f& intersection) {
    CGAL_Segment_2 seg1(toCGALPoint(p1), toCGALPoint(p2));
    CGAL_Segment_2 seg2(toCGALPoint(q1), toCGALPoint(q2));
    auto result = CGAL::intersection(seg1, seg2);
    
    if (result) {
        if (const CGAL_Point_2* p = std::get_if<CGAL_Point_2>(&*result))
        // if (const CGAL_Point_2* p = boost::get<CGAL_Point_2>(&*result))  
        {
            intersection = toCVPoint(*p);
            return true;
        }
    }
    return false;
}






// Function to linearly interpolate height
float interpolateHeight(const cv::Point2f& p1, const cv::Point2f& p2, 
                        float h1, float h2, const cv::Point2f& intersection) {
    float totalDist = cv::norm(p2 - p1);
    float dist1 = cv::norm(intersection - p1);
    float t = dist1 / totalDist;
    return h1 * (1 - t) + h2 * t;
}

// Main function to handle intersection and height interpolation
void handleIntersectionold(const cv::Point2f& firstPoint, const cv::Point2f& secondPoint,
                        const std::vector<double>& distances3D,
                        const cv::Point2f& prevFirstPoint, const cv::Point2f& prevSecondPoint,
                        const std::vector<double>& prevDistances3D,
                        int num_samples) {
    cv::Point2f intersection;
    if (findIntersection(firstPoint, secondPoint, prevFirstPoint, prevSecondPoint, intersection)) {
        // Calculate the position of intersection along the new line segment
        float totalLength = cv::norm(secondPoint - firstPoint);
        float intersectionDist = cv::norm(intersection - firstPoint);
        float t = intersectionDist / totalLength;
        
        // Find the nearby samples
        int index = std::min(static_cast<int>(t * (num_samples - 1)), num_samples - 2);
        
        // Interpolate height for the new line segment
        float h1 = distances3D[index];
        float h2 = distances3D[index + 1];
        cv::Point2f p1 = firstPoint + (secondPoint - firstPoint) * (static_cast<float>(index) / (num_samples - 1));
        cv::Point2f p2 = firstPoint + (secondPoint - firstPoint) * (static_cast<float>(index + 1) / (num_samples - 1));
        float newHeight = interpolateHeight(p1, p2, h1, h2, intersection);
        
        // Repeat for the previous line segment
        float prevTotalLength = cv::norm(prevSecondPoint - prevFirstPoint);
        float prevIntersectionDist = cv::norm(intersection - prevFirstPoint);
        float prevT = prevIntersectionDist / prevTotalLength;
        int prevIndex = std::min(static_cast<int>(prevT * (num_samples - 1)), num_samples - 2);
        
        float prevH1 = prevDistances3D[prevIndex];
        float prevH2 = prevDistances3D[prevIndex + 1];
        cv::Point2f prevP1 = prevFirstPoint + (prevSecondPoint - prevFirstPoint) * (static_cast<float>(prevIndex) / (num_samples - 1));
        cv::Point2f prevP2 = prevFirstPoint + (prevSecondPoint - prevFirstPoint) * (static_cast<float>(prevIndex + 1) / (num_samples - 1));
        float prevHeight = interpolateHeight(prevP1, prevP2, prevH1, prevH2, intersection);
        
        // Now you have newHeight and prevHeight at the intersection point
        std::cout << "Intersection found at: (" << intersection.x << ", " << intersection.y << ")" << std::endl;
        std::cout << "New segment height at intersection: " << newHeight << std::endl;
        std::cout << "Previous segment height at intersection: " << prevHeight << std::endl;
        
        // You can decide how to use these heights (e.g., average them, use the higher one, etc.)
    }
}

struct interpoint{
    cv::Point2f intersectionPoint;
    double prevheight;
    double newheight;
};


void handleIntersection(const cv::Point2f& firstPoint, const cv::Point2f& secondPoint,
                        const std::vector<cv::Point2f>& samples,
                        const std::vector<double>& distances3D,
                        const PolylineData& prevData, interpoint& ip) {
    cv::Point2f intersection;
    if (findIntersection(firstPoint, secondPoint, prevData.firstPoint, prevData.secondPoint, intersection)) {
        // Find the nearest sample points for the new line segment
        auto it = std::min_element(samples.begin(), samples.end(),
            [&intersection](const cv::Point2f& a, const cv::Point2f& b) {
                return cv::norm(a - intersection) < cv::norm(b - intersection);
            });
        int index = std::distance(samples.begin(), it);
        
        // Ensure we don't go out of bounds
        int nextIndex = std::min(index + 1, static_cast<int>(samples.size()) - 1);
        
        // Interpolate height for the new line segment
        float h1 = distances3D[index];
        float h2 = distances3D[nextIndex];
        float newHeight = interpolateHeight(samples[index], samples[nextIndex], h1, h2, intersection);
        
        // Find the nearest sample points for the previous line segment
        auto prevIt = std::min_element(prevData.samples.begin(), prevData.samples.end(),
            [&intersection](const cv::Point2f& a, const cv::Point2f& b) {
                return cv::norm(a - intersection) < cv::norm(b - intersection);
            });
        int prevIndex = std::distance(prevData.samples.begin(), prevIt);
        
        // Ensure we don't go out of bounds
        int prevNextIndex = std::min(prevIndex + 1, static_cast<int>(prevData.samples.size()) - 1);
        
        // Interpolate height for the previous line segment
        float prevH1 = prevData.distance3D[prevIndex];
        float prevH2 = prevData.distance3D[prevNextIndex];
        float prevHeight = interpolateHeight(prevData.samples[prevIndex], prevData.samples[prevNextIndex], prevH1, prevH2, intersection);
        
        // Output the results
        std::cout << "Intersection found at: (" << intersection.x << ", " << intersection.y << ")" << std::endl;
        std::cout << "New segment height at intersection: " << newHeight << std::endl;
        std::cout << "Previous segment height at intersection: " << prevHeight << std::endl;
    ip.intersectionPoint=intersection;
    ip.prevheight=prevHeight;
    ip.newheight=newHeight;



        // You can decide how to use these heights (e.g., average them, use the higher one, etc.)
    }
}


void insertIntersectionPoint(std::vector<cv::Point2f>& samples2D, 
                             std::vector<double>& distance3D,
                             const cv::Point2f& firstPoint, 
                             const cv::Point2f& intersection,
                             double intersectionHeight) {
    // Calculate the distance of the intersection point from the first point
    float intersectionDistance = cv::norm(intersection - firstPoint);

    // Find the correct position to insert the intersection point
    auto it = std::lower_bound(samples2D.begin(), samples2D.end(), intersection,
        [&firstPoint](const cv::Point2f& sample, const cv::Point2f& intersect) {
            return cv::norm(sample - firstPoint) < cv::norm(intersect - firstPoint);
        });

    // Calculate the index where we'll insert
    size_t insertIndex = std::distance(samples2D.begin(), it);

    // Insert the intersection point at the correct position
    samples2D.insert(it, intersection);

    // Insert the intersection height at the same index in distance3D
    distance3D.insert(distance3D.begin() + insertIndex, intersectionHeight);
}


void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        if (!drawingPolyline)
        {
            if (firstPoint.x == -1) // First click
            {
                firstPoint = cv::Point(x, y);
                savedFirstPoint=firstPoint;
                cv::circle(displayImage, firstPoint, 5, cv::Scalar(0, 0, 255), -1);
				// std::cout<<"Drew circle";
            }
            else // Second click
            {
                // cv::Point secondPoint(x, y);
                secondPoint=cv::Point(x,y);
                savedSecondPoint=secondPoint;
                // Check if the line is completely inside the flood-filled region
                cv::LineIterator it(floodRegion, firstPoint, secondPoint, 8);
                bool isInside = true;
                for (int i = 0; i < it.count; i++, ++it)
                {
                    if (!(floodRegion.at<uchar>(it.pos()) == 0))
                    {
                        isInside = false;
                        break;
                    }
                }
                if (isInside)
                {
                    cv::line(displayImage, firstPoint, secondPoint, cv::Scalar(0, 255, 0), 2);
                    drawingPolyline = true;
                }
                
                firstPoint = cv::Point(-1, -1);
            }
        }
        else // Start drawing polyline
        {
            polylinePoints.clear();
            polylinePoints.push_back(cv::Point(x, y));
            isDrawing = true;
        }
    }
    else if (event == cv::EVENT_MOUSEMOVE && isDrawing)
    {
        cv::Mat tempImage = displayImage.clone();
        polylinePoints.push_back(cv::Point(x, y));
        cv::polylines(displayImage, polylinePoints, false, cv::Scalar(255, 0, 0), 2);
        cv::imshow("sketchcross", tempImage);
    }
    else if (event == cv::EVENT_LBUTTONUP && isDrawing)
    {
        polylinePoints.push_back(cv::Point(x, y));
        if(cv::norm(polylinePoints.front()-savedSecondPoint)<cv::norm(polylinePoints.front()-savedFirstPoint))
        if(cv::norm(polylinePoints.back()-savedFirstPoint)<cv::norm(polylinePoints.back()-savedSecondPoint))
     std::reverse(polylinePoints.begin(), polylinePoints.end());
         isDrawing = false;
    cv::polylines(displayImage, polylinePoints, false, cv::Scalar(255, 0, 0), 2);
    cv::Point2f l1_start(polylinePoints.front());
    cv::Point2f l1_end(polylinePoints.back());
    cv::line(displayImage, l1_start, l1_end, cv::Scalar(0, 255, 255), 1);
    cv::Point2f l1_dir = l1_end - l1_start;
    float l1_length = cv::norm(l1_dir);
    l1_dir /= l1_length; // Normalize
    cv::Point2f l2_start(savedFirstPoint);
    cv::Point2f l2_end(savedSecondPoint);
    cv::line(displayImage, l2_start, l2_end, cv::Scalar(255, 255, 0), 1);
    cv::Point2f l2_dir = l2_end - l2_start;
    float l2_length = cv::norm(l2_dir);
    l2_dir /= l2_length; // Normalize
    cv::Point2f perp_dir(-l1_dir.y, l1_dir.x);
    int num_samples = 10; // Adjust as needed
    float perp_length = 1000.0f; // Extended length for intersection check
std::ofstream objFile("polyline3D"+std::to_string(savedPolylines.size())+".obj");
std::vector<cv::Point3f> polyline3D;
std::vector<cv::Point2f> samples2D;
std::vector<double> distance3D;
cv::Point3f point3D(0,0,0);
for (int i = 0; i <= num_samples; ++i)
    {
        float t = static_cast<float>(i) / num_samples;
        cv::Point2f l1_sample = l1_start + t * l1_length * l1_dir;
        cv::Point2f l2_sample = l2_start + t * l2_length * l2_dir;
        cv::Point2f perp_start = l1_sample - perp_length * perp_dir;
        cv::Point2f perp_end = l1_sample + perp_length * perp_dir;
        cv::Point2f intersection = l1_sample;
        float min_distance = std::numeric_limits<float>::max();
        for (size_t j = 0; j < polylinePoints.size() - 1; ++j)
        {
            cv::Point2f p1(polylinePoints[j]), p2(polylinePoints[j + 1]);
            cv::Point2f temp_intersection;
            if (lineLineIntersection(perp_start, perp_end, p1, p2, temp_intersection))
            {
                float distance = cv::norm(temp_intersection - l1_sample);
                if (distance < min_distance)
                {
                    min_distance = distance;
                    intersection = temp_intersection;
                }
            }
        }
        samples2D.push_back(l2_sample);
    float distance = cv::norm(intersection - l2_sample);
           cv::line(displayImage, l2_sample, intersection, cv::Scalar(0, 255, 0), 1);

   point3D=  cv::Point3f(l2_sample.x, l2_sample.y, distance);
    polyline3D.push_back(point3D);
    distance3D.push_back(distance);


        cv::imshow("sketchcross", displayImage);
        drawingPolyline = false;
    }


bool intersects = false;
std::vector<interpoint> ips;
for (const auto& prevData : savedPolylines)
    if (lineIntersects(savedFirstPoint, savedSecondPoint, prevData.firstPoint, prevData.secondPoint)) {
        intersects = true;
                interpoint ip;
                handleIntersection(savedFirstPoint, savedSecondPoint, samples2D, distance3D, prevData,ip);
insertIntersectionPoint(samples2D, distance3D, firstPoint, ip.intersectionPoint,ip.prevheight);
ips.push_back(ip);
    }


if(ips.size()>0)
{

std::vector<cv::Point3f> constraints,newpositions;

for(int i=0;i<ips.size();i++)
{
constraints.push_back(cv::Point3f(ips.at(i).intersectionPoint.x,ips.at(i).intersectionPoint.y,ips.at(i).prevheight));
newpositions.push_back(cv::Point3f(ips.at(i).intersectionPoint.x,ips.at(i).intersectionPoint.y,ips.at(i).newheight));
}

// std::vector<int> constrained_indices;
std::vector<Eigen::Vector3d> eigen_new_positions;

Eigen::MatrixXd V(samples2D.size(), 3);
for (int i = 0; i < samples2D.size(); i++) {
    V.row(i) << samples2D[i].x, samples2D[i].y, distance3D[i];
}
Eigen::MatrixXi E(samples2D.size() - 1, 2);
for (int i = 0; i < samples2D.size() - 1; i++) {
    E.row(i) << i, i + 1;
}
std::vector<int> constrained_indices;
std::vector<Eigen::Vector3d> new_positions;

// Add your constrained vertices here, for example:
constrained_indices.push_back(0);  // First vertex
constrained_indices.push_back(samples2D.size() - 1);  // Last vertex

// Add new positions for constrained vertices
eigen_new_positions.push_back(Eigen::Vector3d(samples2D[0].x, samples2D[0].y, distance3D[0]));
eigen_new_positions.push_back(Eigen::Vector3d(samples2D.back().x, samples2D.back().y, distance3D.back()));

// Add other constrained vertices and their new positions here


// Add other constrained vertices and their new positions
for (int i = 0; i < constraints.size(); i++) {
    // Find the index of the constraint point in samples2D
    auto it = std::find_if(samples2D.begin(), samples2D.end(),
        [&](const cv::Point2f& p) {
            return std::abs(p.x - constraints[i].x) < 1e-6 && std::abs(p.y - constraints[i].y) < 1e-6;
        });
    
    if (it != samples2D.end()) {
        int index = std::distance(samples2D.begin(), it);
        constrained_indices.push_back(index);
        eigen_new_positions.push_back(Eigen::Vector3d(newpositions[i].x, newpositions[i].y, newpositions[i].z));
    }
}

Eigen::VectorXi b = Eigen::Map<Eigen::VectorXi>(constrained_indices.data(), constrained_indices.size());
Eigen::MatrixXd bc(constrained_indices.size(), 3);
for (int i = 0; i < constrained_indices.size(); i++) {
    bc.row(i) = eigen_new_positions[i];
}






Eigen::MatrixXd U;
laplacian_polyline_deform(V, E, b, bc, U);
for (int i = 0; i < samples2D.size(); i++) {
    samples2D[i] = cv::Point2f(U(i, 0), U(i, 1));
    distance3D[i] = U(i, 2);
}

}











    PolylineData datap;
    datap.firstPoint = savedFirstPoint;
    datap.secondPoint = savedSecondPoint;
    datap.polyline3D = polyline3D;
    datap.samples=samples2D;
    datap.distance3D=distance3D;
    savedPolylines.push_back(datap);
    std::cout<<"Inserted ";
    for(int i=0;i<samples2D.size();i++){
     point3D=  cv::Point3f(samples2D.at(i).x,samples2D.at(i).y, distance3D.at(i));
    objFile << "v " << point3D.x << " " << point3D.y << " " << point3D.z << "\n";
    }
    for (size_t i = 0; i < samples2D.size() - 1; ++i)
    objFile << "l " << (i + 1) << " " << (i + 2) << "\n";
 


    }
    cv::imshow("sketchcross", displayImage);
}




int main() {
    cv::Mat image = cv::imread("input.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }
    cv::Mat edges;
    cv::Canny(image, edges, 100, 200);
    cv::Mat mask = cv::Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
    cv::floodFill(image, mask, cv::Point(0, 0), 255, 0, cv::Scalar(), cv::Scalar(), 4 + cv::FLOODFILL_MASK_ONLY);
    cv::Mat floodFilledRegion = mask(cv::Rect(1, 1, image.cols, image.rows));
    cv::Mat boundary;
    cv::Canny(floodFilledRegion*255, boundary, 100, 200);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(boundary, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    int largest_contour_index = -1;
    double largest_area = 0;
    for (int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > largest_area) {
            largest_area = area;
            largest_contour_index = i;
        }
    }
    std::vector<cv::Point> polyline;
    if (!contours.empty()) {
        double epsilon = 0.0002 * cv::arcLength(contours[largest_contour_index], true); // Adjust epsilon for accuracy
        cv::approxPolyDP(contours[largest_contour_index], polyline, epsilon, true);
    }
	floodRegion=floodFilledRegion.clone();
	cv::imshow("floodRegion",floodRegion);
    cv::Mat polylineImage = cv::Mat::zeros(boundary.size(), CV_8UC3);
    cv::polylines(polylineImage, polyline, true, cv::Scalar(0, 255, 0), 1);
    cv::imshow("Polyline Representation", polylineImage);
cv::Mat floodFilledRegion1 = floodFilledRegion.clone()*255; // Deep copy
cv::bitwise_not(floodFilledRegion, floodFilledRegion);
    float radius = 2.0f; // Minimum distance between samples
    std::vector<cv::Point> samples = gridSampling(floodFilledRegion, radius, 1000);
    cv::Mat sampledImage = cv::Mat::zeros(image.size(), CV_8UC3);
    for (const auto& point : samples)
        cv::circle(sampledImage, point, 2, cv::Scalar(0, 0, 255), -1); // Red points
    cv::imshow("Flood-Filled Region", floodFilledRegion); // Scale to make it visible
    cv::imshow("Sampled Points", sampledImage);
    std::vector<Point> cgal_points;
    for (const auto& p : polyline)
        cgal_points.emplace_back(p.x, p.y);
    CDT_plus cdt;
    for (size_t i = 0; i < cgal_points.size(); ++i)
        cdt.insert_constraint(cgal_points[i], cgal_points[(i + 1) % cgal_points.size()]);
    for (const auto& p : samples)
        cdt.insert(Point(p.x, p.y));
    for (CDT_plus::Finite_edges_iterator e = cdt.finite_edges_begin(); e != cdt.finite_edges_end(); ++e) {
        CDT_plus::Segment seg = cdt.segment(*e);
        cv::line(polylineImage, 
                 cv::Point(CGAL::to_double(seg.source().x()), CGAL::to_double(seg.source().y())),
                 cv::Point(CGAL::to_double(seg.target().x()), CGAL::to_double(seg.target().y())),
                 cv::Scalar(255, 0, 0), 1);
    }
 cv::polylines(polylineImage, polyline, true, cv::Scalar(0, 255, 0), 1);
    cv::Mat filteredTriangulation = cv::Mat::zeros(floodFilledRegion1.size(), CV_8UC3);
    for (CDT_plus::Finite_faces_iterator fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) {
        Point p1 = fit->vertex(0)->point();
        Point p2 = fit->vertex(1)->point();
        Point p3 = fit->vertex(2)->point();
        cv::Point cvp1(CGAL::to_double(p1.x()), CGAL::to_double(p1.y()));
        cv::Point cvp2(CGAL::to_double(p2.x()), CGAL::to_double(p2.y()));
        cv::Point cvp3(CGAL::to_double(p3.x()), CGAL::to_double(p3.y()));
        cv::Point centroid = (cvp1 + cvp2 + cvp3) * (1.0/3.0);
        if (!(floodFilledRegion1.at<uchar>(centroid) > 0)) {
            cv::line(filteredTriangulation, cvp1, cvp2, cv::Scalar(255, 255, 255), 1);
            cv::line(filteredTriangulation, cvp2, cvp3, cv::Scalar(255, 255, 255), 1);
            cv::line(filteredTriangulation, cvp3, cvp1, cv::Scalar(255, 255, 255), 1);
        }
    }
	displayImage=filteredTriangulation.clone();
    cv::namedWindow("sketchcross");
    cv::setMouseCallback("sketchcross", mouseCallback, nullptr);
    cv::imshow("sketchcross", filteredTriangulation);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}