# CMath




CMath is set of C++ classes for Vector and Matrix algebra used in computer graphics and relativity physics . The library consits of these classes:


    
    Complex - two dimensional complex number for 2D 
    Vector2 - two dimensional vector for 2D vertices and texture coordinates
    Vector3 - three dimensional vector for 3D vertices, normals and texture coordinates and also for color
    Vector4 - four dimensional vector for 3D vertices and 4D vecrtices, normals, texture coordinates and color with alpha channell.
    LoretzVector - four dimensional vector for 4D relativity physics 
    Matrix3 - matrix 3x3 for rotation (used in ODE)
    Matrix4 - matrix 4x4 for general geometrix transformations
    Quaternion - quaternion re 3x im 1x, for rotation_3D
    Octonion - Octonion re 7x im 1x, for rotation_6D
   
    
    
Note that this library is set of C++ class that has all(!) method inlined. (for performance reasons)


Features

    basic aritemetic operations - using operators
    basic linear algebra operations - such as transpose, dot product, etc.
    aliasis for vertex coordinates - it means:
    Vector3f v;
    // use vertex coordinates
    v.x = 1; v.y = 2; v.z = -1;
    // use texture coordinates
    v.s = 0; v.t = 1; v.u = 0.5;
    // use color coordinates
    v.r = 1; v.g = 0.5; v.b = 0;
    conversion constructor and assign operators - so you can assign a value of rpVector3D<T1> type to a variable of rpVector3D<T2> type for any convertable T1, T2 type pairs. In other words, you can do this:
    Vector3f f3; Vector3d d3 = f3;
    ...
    f3 = d3;

    
Status

    Classes rpVector2D<T>, rpVector3D<T>, rpVector4D<T> are supposed to be stable. I have been using these libraries for two or three years.
    Classes rpMatrix3x3<T>, rpMatrix4x4<T> were tested for barely all operations and seems to be everything OK.
    Class rpQuaternion<T> was tested for barely all operations and seems to be good.

    
Tricks

You can pass vector or matrix class directly as argument appropriate OpenGL function,

	Vector2f t;
	Vector3f n,v;
	Matrix4f transform;
	
	glMultiMatrixf(transform);
	
	glTexCoord2fv(t);
	glNormal3fv(n);
	glVertex3fv(v);
