#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <cmath>
#include <GLFW/glfw3.h>

// Define drawing states.
enum DrawState { WAIT_FOR_ENDPOINTS, DRAW_STROKE };
DrawState current_state = WAIT_FOR_ENDPOINTS;

// Flags for endpoint input.
bool first_endpoint_set = false;
bool endpointsComplete = false;  // true when both endpoints are set

// Endpoints of the 2D line segment.
Eigen::RowVector2d p0, p1;

// Stroke storage.
std::vector<Eigen::RowVector2d> currentStroke;

// Global flag for camera rotation lock (true = locked, i.e. drawing enabled).
bool rotationLocked = true;

// Storage for previously drawn curves.
std::vector<Eigen::MatrixXd> curves2D; // uniform samples in 2D
std::vector<Eigen::MatrixXd> curves3D; // corresponding 3D curves

// Helper: update camera view based on all stored 3D curve vertices.
void updateCamera(igl::opengl::glfw::Viewer &viewer)
{
  if(curves3D.empty())
    return;
  int totalRows = 0;
  for (const auto &curve : curves3D)
    totalRows += curve.rows();
  Eigen::MatrixXd V(totalRows, 3);
  int startRow = 0;
  for (const auto &curve : curves3D)
  {
    int r = curve.rows();
    V.block(startRow, 0, r, 3) = curve;
    startRow += r;
  }
  viewer.core().align_camera_center(V);
}

// Uniformly sample a 2D line segment between p0 and p1.
Eigen::MatrixXd sampleLine2D(const Eigen::RowVector2d &p0,
                             const Eigen::RowVector2d &p1,
                             int numSamples)
{
  Eigen::MatrixXd samples(numSamples, 2);
  for (int i = 0; i < numSamples; ++i)
  {
    double t = double(i) / (numSamples - 1);
    samples.row(i) = (1 - t) * p0 + t * p1;
  }
  return samples;
}

// Convert uniformly sampled 2D points into a 3D curve.
// The z-coordinate is computed as the Euclidean distance between the
// corresponding stroke point and the uniform sample.
Eigen::MatrixXd convertTo3DCurve(const Eigen::MatrixXd &lineSamples2D,
                                 const Eigen::MatrixXd &stroke2D)
{
  int n = lineSamples2D.rows();
  Eigen::MatrixXd curve3D(n, 3);
  for (int i = 0; i < n; ++i)
  {
    curve3D(i, 0) = lineSamples2D(i, 0);
    curve3D(i, 1) = lineSamples2D(i, 1);
    curve3D(i, 2) = (lineSamples2D.row(i) - stroke2D.row(i)).norm();
  }
  return curve3D;
}

// Simple 2D segment intersection test between segments (p0,p1) and (p2,p3).
bool intersectSegments(const Eigen::RowVector2d &p0, const Eigen::RowVector2d &p1,
                       const Eigen::RowVector2d &p2, const Eigen::RowVector2d &p3,
                       Eigen::RowVector2d &intersection)
{
  Eigen::RowVector2d s1 = p1 - p0;
  Eigen::RowVector2d s2 = p3 - p2;
  
  double denom = -s2(0) * s1(1) + s1(0) * s2(1);
  if (std::abs(denom) < 1e-8)
    return false;
  
  double s = (-s1(1) * (p0(0) - p2(0)) + s1(0) * (p0(1) - p2(1))) / denom;
  double t = ( s2(0) * (p0(1) - p2(1)) - s2(1) * (p0(0) - p2(0))) / denom;
  
  if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
  {
    intersection = p0 + t * s1;
    return true;
  }
  return false;
}

// ARAP deformation for a 1D chain (curve) in 3D.
//   X0: initial guess for the deformed curve (n x 3)
//   base: the original (rest) shape (n x 3) whose edge differences we want to preserve
//   constraints: indices of vertices to be fixed (e.g., {0, k, n-1})
//   targetPositions: corresponding target positions (c x 3)
//   maxIter: maximum number of iterations
//   tol: convergence tolerance
Eigen::MatrixXd deformCurveARAP(const Eigen::MatrixXd &X0,
                                const Eigen::MatrixXd &base,
                                const std::vector<int> &constraints,
                                const Eigen::MatrixXd &targetPositions,
                                int maxIter = 10,
                                double tol = 1e-4)
{
  int n = X0.rows();
  // X will hold our current deformed positions; initialize with X0.
  Eigen::MatrixXd X = X0;

  // Build the Laplacian matrix L for a chain (n x n):
  //   For vertex 0: L(0,0)=1, L(0,1)=-1.
  //   For vertices 1..n-2: L(i,i-1)=-1, L(i,i)=2, L(i,i+1)=-1.
  //   For vertex n-1: L(n-1,n-2)=-1, L(n-1,n-1)=1.
  Eigen::SparseMatrix<double> L(n, n);
  std::vector<Eigen::Triplet<double>> tripL;
  if(n >= 1) {
    tripL.push_back(Eigen::Triplet<double>(0, 0, 1));
    if(n > 1)
      tripL.push_back(Eigen::Triplet<double>(0, 1, -1));
    for(int i = 1; i < n - 1; i++){
      tripL.push_back(Eigen::Triplet<double>(i, i - 1, -1));
      tripL.push_back(Eigen::Triplet<double>(i, i, 2));
      tripL.push_back(Eigen::Triplet<double>(i, i + 1, -1));
    }
    if(n > 1) {
      int i = n - 1;
      tripL.push_back(Eigen::Triplet<double>(i, i - 1, -1));
      tripL.push_back(Eigen::Triplet<double>(i, i, 1));
    }
  }
  L.setFromTriplets(tripL.begin(), tripL.end());

  // Set up the constraint matrix.
  // For each constraint vertex index, we have a row in C that picks that vertex.
  int numConstraints = constraints.size();
  Eigen::SparseMatrix<double> C(numConstraints, n);
  std::vector<Eigen::Triplet<double>> tripC;
  for(int i = 0; i < numConstraints; i++){
    int idx = constraints[i];
    tripC.push_back(Eigen::Triplet<double>(i, idx, 1));
  }
  C.setFromTriplets(tripC.begin(), tripC.end());
  
  // We enforce constraints strongly by adding them to the system with a large weight.
  double W = 1e6;
  int m = L.rows() + C.rows(); // total rows after augmentation

  // Build the augmented system matrix A = [ L ; sqrt(W)*C ].
  Eigen::SparseMatrix<double> A(m, n);
  std::vector<Eigen::Triplet<double>> tripA;
  for (int k = 0; k < L.outerSize(); k++){
    for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it){
      tripA.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
    }
  }
  for (int k = 0; k < C.outerSize(); k++){
    for (Eigen::SparseMatrix<double>::InnerIterator it(C, k); it; ++it){
      tripA.push_back(Eigen::Triplet<double>(L.rows() + it.row(), it.col(), std::sqrt(W) * it.value()));
    }
  }
  A.setFromTriplets(tripA.begin(), tripA.end());

  // ARAP Iterations:
  // In the local step, for each edge we compute a rotation that best aligns the edge in the base shape with the current edge.
  // In the global step, we solve a linear system to update the vertex positions.
  int max_iterations = maxIter;
  double prevError = 1e12;
  for (int iter = 0; iter < max_iterations; iter++){
    // Local step: compute per-edge rotations R_i (for i=0,..., n-2).
    std::vector<Eigen::Matrix3d> R(n - 1, Eigen::Matrix3d::Identity());
    for (int i = 0; i < n - 1; i++){
      // a: original edge from the base shape.
      Eigen::Vector3d a = (base.row(i + 1) - base.row(i)).transpose();
      // b: current deformed edge.
      Eigen::Vector3d b = (X.row(i + 1) - X.row(i)).transpose();
      double norm_a = a.norm();
      double norm_b = b.norm();
      if (norm_a < 1e-8 || norm_b < 1e-8) {
        R[i] = Eigen::Matrix3d::Identity();
      } else {
        double cosTheta = a.dot(b) / (norm_a * norm_b);
        if (cosTheta > 1.0) cosTheta = 1.0;
        if (cosTheta < -1.0) cosTheta = -1.0;
        double angle = acos(cosTheta);
        Eigen::Vector3d axis = a.cross(b);
        if (axis.norm() < 1e-8) {
          R[i] = Eigen::Matrix3d::Identity();
        } else {
          axis.normalize();
          R[i] = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
        }
      }
    }

    // Global step: compute the divergence vector r (n x 3) from the rotations.
    // For a chain:
    //   r(0) = R_0 * (base[1]-base[0])
    //   For i=1,..,n-2: r(i) = R_{i-1}*(base[i]-base[i-1]) + R_i*(base[i+1]-base[i])
    //   r(n-1) = R_{n-2}*(base[n-1]-base[n-2])
    Eigen::MatrixXd r = Eigen::MatrixXd::Zero(n, 3);
    if (n > 1) {
      Eigen::Vector3d a0 = (base.row(1) - base.row(0)).transpose();
      r.row(0) = (R[0] * a0).transpose();
    }
    for (int i = 1; i < n - 1; i++){
      Eigen::Vector3d a_prev = (base.row(i) - base.row(i - 1)).transpose();
      Eigen::Vector3d a_curr = (base.row(i + 1) - base.row(i)).transpose();
      r.row(i) = (R[i - 1] * a_prev + R[i] * a_curr).transpose();
    }
    if (n > 1) {
      Eigen::Vector3d a_last = (base.row(n - 1) - base.row(n - 2)).transpose();
      r.row(n - 1) = (R[n - 2] * a_last).transpose();
    }
    
    // Build augmented right-hand side b_aug for each coordinate:
    //   b_aug = [ r(:,dim) ; sqrt(W)*targetPositions(:,dim) ]
    Eigen::MatrixXd X_new = X;
    double totalError = 0.0;
    for (int dim = 0; dim < 3; dim++){
      Eigen::VectorXd b_aug(m);
      for (int i = 0; i < L.rows(); i++){
        b_aug(i) = r(i, dim);
      }
      for (int i = 0; i < C.rows(); i++){
        b_aug(L.rows() + i) = std::sqrt(W) * targetPositions(i, dim);
      }
      // Solve the normal equations: (A^T A) x = A^T b_aug.
      Eigen::SparseMatrix<double> AtA(n, n);
      AtA = A.transpose() * A;
      Eigen::VectorXd Atb = A.transpose() * b_aug;
      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
      solver.compute(AtA);
      if (solver.info() != Eigen::Success) {
        std::cerr << "Decomposition failed in global step!" << std::endl;
        return X;
      }
      Eigen::VectorXd x_sol = solver.solve(Atb);
      if (solver.info() != Eigen::Success) {
        std::cerr << "Solving failed in global step!" << std::endl;
        return X;
      }
      X_new.col(dim) = x_sol;
    }
    
    // Check for convergence.
    double error = (X_new - X).norm();
    totalError = error;
    if (error < tol)
    {
      X = X_new;
      break;
    }
    X = X_new;
  }
  return X;
}

// Update the viewer with all stored 3D curves (drawn in red).
void updateViewer(igl::opengl::glfw::Viewer &viewer)
{
  viewer.data().clear();
  for (const auto &curve : curves3D) {
    int n = curve.rows();
    for (int i = 0; i < n - 1; i++) {
      Eigen::RowVector3d a = curve.row(i);
      Eigen::RowVector3d b = curve.row(i+1);
      viewer.data().add_edges(a, b, Eigen::RowVector3d(1, 0, 0));
    }
  }
}

int main(int argc, char *argv[])
{
  igl::opengl::glfw::Viewer viewer;
  viewer.core().orthographic = true;
  
  // --- Key Callbacks ---
  viewer.callback_key_down = [&](igl::opengl::glfw::Viewer &viewer, unsigned int key, int modifiers) -> bool {
    // Toggle camera rotation lock with 'R'
    if(key == GLFW_KEY_R) {
      rotationLocked = !rotationLocked;
      if(!rotationLocked) {  // if unlocked, cancel any drawing in progress
        current_state = WAIT_FOR_ENDPOINTS;
        first_endpoint_set = false;
        endpointsComplete = false;
        currentStroke.clear();
        std::cout << "Camera rotation unlocked. Stroke drawing canceled." << std::endl;
      } else {
        std::cout << "Camera rotation locked. Drawing enabled." << std::endl;
      }
      return true;
    }
    // Press 'N' to start a new segment.
    if(key == GLFW_KEY_N) {
      current_state = WAIT_FOR_ENDPOINTS;
      first_endpoint_set = false;
      endpointsComplete = false;
      currentStroke.clear();
      std::cout << "New segment: click to set endpoints." << std::endl;
      return true;
    }
    return true; // consume other key events in our app
  };
  
  // --- Pre-draw Callback ---
  viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &viewer) -> bool {
    updateViewer(viewer);
    // Show endpoints in blue.
    if(first_endpoint_set)
      viewer.data().add_points(Eigen::RowVector3d(p0(0), p0(1), 0), Eigen::RowVector3d(0, 0, 1));
    if(endpointsComplete) {
      viewer.data().add_points(Eigen::RowVector3d(p1(0), p1(1), 0), Eigen::RowVector3d(0, 0, 1));
      viewer.data().add_edges(Eigen::RowVector3d(p0(0), p0(1), 0),
                              Eigen::RowVector3d(p1(0), p1(1), 0),
                              Eigen::RowVector3d(0, 0, 1));
      std::cout << "Endpoints set. Press and hold the mouse to draw the stroke." << std::endl;
    }
    return false;
  };
  
  // --- Mouse Callbacks ---
  // Mouse-down:
  // * In WAIT_FOR_ENDPOINTS: first click sets p0; second click sets p1.
  // * If endpoints are complete, the next mouse-down initiates stroke drawing.
  viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer &viewer, int button, int modifier) -> bool {
    // If camera rotation is unlocked, skip drawing.
    if(!rotationLocked)
      return false;
      
    double x = viewer.current_mouse_x;
    double y = viewer.current_mouse_y;
    Eigen::Vector3d win;
    win << x, viewer.core().viewport(3) - y, 0.5;
    Eigen::Vector3d world;
    igl::unproject(win, viewer.core().view, viewer.core().proj, viewer.core().viewport, world);
    Eigen::RowVector2d pos(world[0], world[1]);
    
    if(current_state == WAIT_FOR_ENDPOINTS) {
      if(!first_endpoint_set) {
        p0 = pos;
        first_endpoint_set = true;
        std::cout << "First endpoint set: " << p0 << std::endl;
      } else if(!endpointsComplete) {
        p1 = pos;
        endpointsComplete = true;
        std::cout << "Second endpoint set: " << p1 << std::endl;
        std::cout << "Now press and hold the mouse to draw the stroke." << std::endl;
      }
      else {
        // If endpoints are already complete, start drawing stroke.
        current_state = DRAW_STROKE;
        currentStroke.clear();
        std::cout << "Stroke drawing started." << std::endl;
      }
    }
    return true; // consume event if drawing is enabled
  };
  
  // Mouse-move: record stroke points if in DRAW_STROKE mode.
  viewer.callback_mouse_move = [&](igl::opengl::glfw::Viewer &viewer, int mouse_x, int mouse_y) -> bool {
    // If not in drawing mode or if rotation is unlocked, do nothing.
    if(current_state != DRAW_STROKE || !rotationLocked)
      return false;
      
    double x = viewer.current_mouse_x;
    double y = viewer.current_mouse_y;
    Eigen::Vector3d win;
    win << x, viewer.core().viewport(3) - y, 0.5;
    Eigen::Vector3d world;
    igl::unproject(win, viewer.core().view, viewer.core().proj, viewer.core().viewport, world);
    Eigen::RowVector2d pos(world[0], world[1]);
    currentStroke.push_back(pos);
    
    // Live feedback: draw the current stroke in green.
    int n = currentStroke.size();
    if(n >= 2) {
      Eigen::MatrixXd stroke(n,3);
      for (int i = 0; i < n; i++) {
        stroke(i,0) = currentStroke[i](0);
        stroke(i,1) = currentStroke[i](1);
        stroke(i,2) = 0;
      }
      viewer.data().clear();
      updateViewer(viewer);
      for (int i = 0; i < n - 1; i++) {
        Eigen::RowVector3d a = stroke.row(i);
        Eigen::RowVector3d b = stroke.row(i+1);
        viewer.data().add_edges(a, b, Eigen::RowVector3d(0, 1, 0));
      }
      viewer.data().add_points(Eigen::RowVector3d(currentStroke.back()(0), currentStroke.back()(1), 0),
                               Eigen::RowVector3d(0, 1, 0));
      updateCamera(viewer);
    }
    return true;
  };
  
  // Mouse-up: finish stroke drawing and process the new curve.
  viewer.callback_mouse_up = [&](igl::opengl::glfw::Viewer &viewer, int button, int modifier) -> bool {
    // Only process if we are in DRAW_STROKE mode and drawing is enabled.
    if(current_state != DRAW_STROKE || !rotationLocked)
      return false;
      
    int n = currentStroke.size();
    if(n < 2) {
      std::cout << "Stroke too short, ignoring." << std::endl;
      current_state = WAIT_FOR_ENDPOINTS;
      first_endpoint_set = false;
      endpointsComplete = false;
      return true;
    }
    Eigen::MatrixXd stroke2D(n, 2);
    for (int i = 0; i < n; i++)
      stroke2D.row(i) = currentStroke[i];
    
    Eigen::MatrixXd lineSamples = sampleLine2D(p0, p1, n);
    Eigen::MatrixXd curve3D = convertTo3DCurve(lineSamples, stroke2D);
    
    // If a previous curve exists, check for intersection.
    if(!curves2D.empty()) {
      Eigen::RowVector2d firstCurveP0 = curves2D[0].row(0);
      Eigen::RowVector2d firstCurveP1 = curves2D[0].row(curves2D[0].rows()-1);
      Eigen::RowVector2d interPt;
      if(intersectSegments(p0, p1, firstCurveP0, firstCurveP1, interPt)) {
        double t = (interPt - p0).norm() / (p1 - p0).norm();
        int intersectIdx = std::round(t * (n - 1));
        if(intersectIdx < 0) intersectIdx = 0;
        if(intersectIdx >= n) intersectIdx = n - 1;
        
        int firstCurveIdx = std::round(t * (curves3D[0].rows() - 1));
        Eigen::RowVector3d targetInterPt = curves3D[0].row(firstCurveIdx);
        
        std::cout << "Intersection detected at t = " << t << std::endl;
        std::cout << "Intersection index on new curve: " << intersectIdx << std::endl;
        std::cout << "Target intersection point from first curve: " << targetInterPt << std::endl;
        
        std::vector<int> constraints = { 0, intersectIdx, n - 1 };
        Eigen::MatrixXd targetPositions(3, 3);
        targetPositions.row(0) = curve3D.row(0);
        targetPositions.row(1) = targetInterPt;
        targetPositions.row(2) = curve3D.row(n - 1);
        
        curve3D = deformCurveARAP(curve3D,curve3D, constraints, targetPositions);
      }
    }
    
    curves2D.push_back(lineSamples);
    curves3D.push_back(curve3D);
    
    updateViewer(viewer);
    viewer.data().clear_points();
    updateCamera(viewer);
    
    // Reset for next segment.
    current_state = WAIT_FOR_ENDPOINTS;
    first_endpoint_set = false;
    endpointsComplete = false;
    currentStroke.clear();
    std::cout << "Stroke drawing finished. Press 'N' to start a new segment." << std::endl;
    return true;
  };
  
  viewer.launch();
  return 0;
}
