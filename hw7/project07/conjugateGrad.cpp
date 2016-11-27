// implementation for CG on 5 point Laplacian.

#include "conjugateGrad.h"


// multiply the vector u by minus the five point laplacian
// return the product in Cu.

// in the interior
// Cu(i,j) = sq(N-1)*(4u(i,j)-u(i-1.j)-u(i+1,j)-u(i,j-1)-u(i,j+1))

// on the boundary both u and Cu should be zero. A point 
// (i,j) is on the boudary if i is 0 or N-1 or j is 0 or N-1.


// The problem has homogenous Dirichlet boundary values
// so the matrix C has rows that look like the identity
// on the boundary. The vector u should be zero on the
// boundary.

// It is assumed that the mesh is on (0,1)x(0,1) and
// that the mesh is NxN, where N is the number of points
// and N-1 is the number of intervals.

// Note that the action of the  matrix is symmetric on 
// the space of vectors that vanish on the boundary.

void mul5ptLaplace( const Matrix& u, Matrix& Cu )
{
  //???
}

// Take one step of CG for the 5 point laplacian with zero boundary
// values
void stepCG( Matrix& u, Matrix& s, Matrix& r, double& normResSq )
{
  //???
}


// fill the interior of a vector with function values
// set the boundary values to zero
void fill( Matrix& u, pt2dbl f, double dx)
{
  int N = u.Row(), i,j;
  if( N < 3 )
    warning("fill: called with N < 3 ");
  for(i =0; i< N; i++ ){ // zero boundary
    u(i,0) = u(i,N-1) = u(0,i) = u(N-1,i) = 0.0;
  }
  for( i=1; i<N-1; i++ ){ // fill interior
    for(j=1; j<N-1; j++)
      u(i,j) = f(i*dx,j*dx);
  }
}
