#pragma once

#include "Core.h"
#include "Sampling.h"

class Ray
{
public:
	Vec3 o;
	Vec3 dir;
	Vec3 invDir;
	Ray()
	{
	}
	Ray(Vec3 _o, Vec3 _d)
	{
		init(_o, _d);
	}
	void init(Vec3 _o, Vec3 _d)
	{
		o = _o;
		dir = _d;
		invDir = Vec3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
	}
	Vec3 at(const float t) const
	{
		return (o + (dir * t));
	}
};

class Plane
{
public:
	Vec3 n;
	float d;
	void init(Vec3& _n, float _d)
	{
		n = _n;
		d = _d;
	}
	// Add code here
	bool rayIntersect(Ray& r, float& t)
	{
		float denom = n.dot(r.dir);

		if (fabs(denom) < 1e-6)
			return false;

		float num = -(n.dot(r.o) + d);
		t = num / denom;

		return (t >= 0);
	}
};

#define EPSILON 0.001f

class Triangle
{
public:
	Vertex vertices[3];
	Vec3 e1; // Edge 1
	Vec3 e2; // Edge 2
	Vec3 n; // Geometric Normal
	float area; // Triangle area
	float d; // For ray triangle if needed
	unsigned int materialIndex;
	void init(Vertex v0, Vertex v1, Vertex v2, unsigned int _materialIndex)
	{
		materialIndex = _materialIndex;
		vertices[0] = v0;
		vertices[1] = v1;
		vertices[2] = v2;
		e1 = vertices[2].p - vertices[1].p;
		e2 = vertices[0].p - vertices[2].p;
		n = e1.cross(e2).normalize();
		area = e1.cross(e2).length() * 0.5f;
		d = Dot(n, vertices[0].p);
	}
	Vec3 centre() const
	{
		return (vertices[0].p + vertices[1].p + vertices[2].p) / 3.0f;
	}
	// Add code here
	bool rayIntersect(const Ray& r, float& t, float& u, float& v) const
	{
		float denom = Dot(n, r.dir);
		if (denom == 0) { return false; }
		t = (d - Dot(n, r.o)) / denom;
		if (t < 0) { return false; }
		Vec3 p = r.at(t);
		float invArea = 1.0f / Dot(e1.cross(e2), n);
		u = Dot(e1.cross(p - vertices[1].p), n) * invArea;
		if (u < 0 || u > 1.0f) { return false; }
		v = Dot(e2.cross(p - vertices[2].p), n) * invArea;
		if (v < 0 || (u + v) > 1.0f) { return false; }
		return true;
	}
	void interpolateAttributes(const float alpha, const float beta, const float gamma, Vec3& interpolatedNormal, float& interpolatedU, float& interpolatedV) const
	{
		interpolatedNormal = vertices[0].normal * alpha + vertices[1].normal * beta + vertices[2].normal * gamma;
		interpolatedNormal = interpolatedNormal.normalize();
		interpolatedU = vertices[0].u * alpha + vertices[1].u * beta + vertices[2].u * gamma;
		interpolatedV = vertices[0].v * alpha + vertices[1].v * beta + vertices[2].v * gamma;
	}
	// Add code here
	Vec3 sample(Sampler* sampler, float& pdf)
	{
		return Vec3(0, 0, 0);
	}
	Vec3 gNormal()
	{
		return (n * (Dot(vertices[0].normal, n) > 0 ? 1.0f : -1.0f));
	}
};

class AABB
{
public:
	Vec3 max;
	Vec3 min;
	AABB()
	{
		reset();
	}
	void reset()
	{
		max = Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		min = Vec3(FLT_MAX, FLT_MAX, FLT_MAX);
	}
	void extend(const Vec3 p)
	{
		max = Max(max, p);
		min = Min(min, p);
	}
	// Add code here
	bool rayAABB(const Ray& r, float& t)
	{
		float tmin = (min.x - r.o.x) * r.invDir.x;
		float tmax = (max.x - r.o.x) * r.invDir.x;
		if (tmin > tmax) std::swap(tmin, tmax);

		float tymin = (min.y - r.o.y) * r.invDir.y;
		float tymax = (max.y - r.o.y) * r.invDir.y;
		if (tymin > tymax) std::swap(tymin, tymax);

		if ((tmin > tymax) || (tymin > tmax))
			return false;

		if (tymin > tmin)
			tmin = tymin;
		if (tymax < tmax)
			tmax = tymax;

		float tzmin = (min.z - r.o.z) * r.invDir.z;
		float tzmax = (max.z - r.o.z) * r.invDir.z;
		if (tzmin > tzmax) std::swap(tzmin, tzmax);

		if ((tmin > tzmax) || (tzmin > tmax))
			return false;

		if (tzmin > tmin)
			tmin = tzmin;
		if (tzmax < tmax)
			tmax = tzmax;

		t = tmin;
		return true;
	}
	// Add code here
	bool rayAABB(const Ray& r)
	{
		float t;
		return rayAABB(r, t);
	}
	// Add code here
	float area()
	{
		Vec3 size = max - min;
		return ((size.x * size.y) + (size.y * size.z) + (size.x * size.z)) * 2.0f;
	}
};

class Sphere
{
public:
	Vec3 centre;
	float radius;
	void init(Vec3& _centre, float _radius)
	{
		centre = _centre;
		radius = _radius;
	}
	// Add code here
	bool rayIntersect(Ray& r, float& t)
	{
		Vec3 oc = r.o - centre;
		float a = r.dir.dot(r.dir);
		float b = 2.0f * oc.dot(r.dir);
		float c = oc.dot(oc) - radius * radius;
		float discriminant = b * b - 4 * a * c;

		if (discriminant < 0)
		{
			return false;
		}
		else
		{
			t = (-b - sqrt(discriminant)) / (2.0f * a);
			if (t < 0)
			{
				t = (-b + sqrt(discriminant)) / (2.0f * a);
				if (t < 0)
				{
					return false;
				}
			}
			return true;
		}
	}
};

struct IntersectionData
{
	unsigned int ID;
	float t;
	float alpha;
	float beta;
	float gamma;
};

#define MAXNODE_TRIANGLES 8
#define TRAVERSE_COST 1.0f
#define TRIANGLE_COST 2.0f
#define BUILD_BINS 32

class BVHNode
{
public:
	AABB bounds;
	BVHNode* r;
	BVHNode* l;
	unsigned int offset;
	unsigned int size;
	// This can store an offset and number of triangles in a global triangle list for example
	// But you can store this however you want!
	// unsigned int offset;
	// unsigned char num;
	BVHNode()
	{
		r = NULL;
		l = NULL;
		offset = 0;
		size = 0;
	}
	// Note there are several options for how to implement the build method. Update this as required
	void build(std::vector<Triangle>& inputTriangles, std::vector<Triangle>& outputTriangles)
	{
		// Add BVH building code here
				// Compute the bounding box for all triangles in this node.
		bounds.reset();
		for (const auto& tri : inputTriangles)
		{
			bounds.extend(tri.vertices[0].p);
			bounds.extend(tri.vertices[1].p);
			bounds.extend(tri.vertices[2].p);
		}

		// If the number of triangles is small, create a leaf node.
		if (inputTriangles.size() <= MAXNODE_TRIANGLES)
		{
			offset = outputTriangles.size();
			size = inputTriangles.size();
			outputTriangles.insert(outputTriangles.end(), inputTriangles.begin(), inputTriangles.end());
			return;
		}

		// Choose the splitting axis based on the bounding box extents.
		Vec3 extents = bounds.max - bounds.min;
		int axis = 0;
		if (extents.y > extents.x) axis = 1;
		if (extents.z > extents.coords[axis]) axis = 2;

		// Use SAH with a fixed number of bins.
		const int nBins = BUILD_BINS;
		struct Bin
		{
			AABB bounds;
			int count;
			Bin() : count(0) { bounds.reset(); }
		};
		std::vector<Bin> bins(nBins);

		// Partition triangles into bins based on the centroid along the chosen axis.
		for (const auto& tri : inputTriangles)
		{
			Vec3 center = tri.centre();
			float relativePos = (center.coords[axis] - bounds.min.coords[axis]) / extents.coords[axis];
			int binIdx = std::min(nBins - 1, static_cast<int>(relativePos * nBins));
			bins[binIdx].count++;
			bins[binIdx].bounds.extend(tri.vertices[0].p);
			bins[binIdx].bounds.extend(tri.vertices[1].p);
			bins[binIdx].bounds.extend(tri.vertices[2].p);
		}

		// Compute accumulated data for candidate splits.
		std::vector<int> leftCount(nBins, 0), rightCount(nBins, 0);
		std::vector<AABB> leftBounds(nBins), rightBounds(nBins);
		{
			int count = 0;
			for (int i = 0; i < nBins; ++i)
			{
				count += bins[i].count;
				leftCount[i] = count;
				if (i == 0)
					leftBounds[i] = bins[i].bounds;
				else
				{
					leftBounds[i].min = Min(leftBounds[i - 1].min, bins[i].bounds.min);
					leftBounds[i].max = Max(leftBounds[i - 1].max, bins[i].bounds.max);
				}
			}
		}
		{
			int count = 0;
			for (int i = nBins - 1; i >= 0; --i)
			{
				count += bins[i].count;
				rightCount[i] = count;
				if (i == nBins - 1)
					rightBounds[i] = bins[i].bounds;
				else
				{
					rightBounds[i].min = Min(rightBounds[i + 1].min, bins[i].bounds.min);
					rightBounds[i].max = Max(rightBounds[i + 1].max, bins[i].bounds.max);
				}
			}
		}

		// Evaluate SAH cost for splitting between bins i and i+1.
		float bestCost = FLT_MAX;
		int bestSplit = -1;
		float invTotalArea = 1.0f / bounds.area();
		for (int i = 0; i < nBins - 1; ++i)
		{
			float cost = TRAVERSE_COST +
				(TRIANGLE_COST * ((leftCount[i] * leftBounds[i].area() +
					rightCount[i + 1] * rightBounds[i + 1].area()) * invTotalArea));
			if (cost < bestCost)
			{
				bestCost = cost;
				bestSplit = i;
			}
		}

		// If no beneficial split is found, make this node a leaf.
		if (inputTriangles.size() <= MAXNODE_TRIANGLES || bestCost > TRIANGLE_COST * inputTriangles.size())
		{
			offset = outputTriangles.size();
			size = inputTriangles.size();
			outputTriangles.insert(outputTriangles.end(), inputTriangles.begin(), inputTriangles.end());
			return;
		}

		// Partition the triangles into left and right sets.
		std::vector<Triangle> leftTriangles, rightTriangles;
		float splitPos = bounds.min.coords[axis] + ((bestSplit + 1) / static_cast<float>(nBins)) * extents.coords[axis];
		for (const auto& tri : inputTriangles)
		{
			if (tri.centre().coords[axis] < splitPos)
				leftTriangles.push_back(tri);
			else
				rightTriangles.push_back(tri);
		}

		// Fallback: If one side is empty, use an equal partition.
		if (leftTriangles.empty() || rightTriangles.empty())
		{
			size_t mid = inputTriangles.size() / 2;
			leftTriangles = std::vector<Triangle>(inputTriangles.begin(), inputTriangles.begin() + mid);
			rightTriangles = std::vector<Triangle>(inputTriangles.begin() + mid, inputTriangles.end());
		}

		// Recursively build BVH for child nodes.
		l = new BVHNode();
		r = new BVHNode();
		l->build(leftTriangles, outputTriangles);
		r->build(rightTriangles, outputTriangles);
	}
	void traverse(const Ray& ray, const std::vector<Triangle>& triangles, IntersectionData& intersection)
	{
		// Add BVH Traversal code here
		float tHit;
		if (!bounds.rayAABB(ray, tHit))
			return;

		// If leaf node, test the triangles.
		if (l == nullptr && r == nullptr)
		{
			for (unsigned int i = offset; i < offset + size; ++i)
			{
				float t, u, v;
				if (triangles[i].rayIntersect(ray, t, u, v) && t < intersection.t)
				{
					intersection.t = t;
					intersection.alpha = 1.0f - u - v;
					intersection.beta = u;
					intersection.gamma = v;
					intersection.ID = i;
				}
			}
		}
		else
		{
			if (l) l->traverse(ray, triangles, intersection);
			if (r) r->traverse(ray, triangles, intersection);
		}
	}
	IntersectionData traverse(const Ray& ray, const std::vector<Triangle>& triangles)
	{
		IntersectionData intersection;
		intersection.t = FLT_MAX;
		traverse(ray, triangles, intersection);
		return intersection;
	}
	bool traverseVisible(const Ray& ray, const std::vector<Triangle>& triangles, const float maxT)
	{
		// Add visibility code here
		float tHit;
		if (!bounds.rayAABB(ray, tHit))
			return false;

		// If leaf node, test the triangles.
		if (l == nullptr && r == nullptr)
		{
			for (unsigned int i = offset; i < offset + size; ++i)
			{
				float t, u, v;
				if (triangles[i].rayIntersect(ray, t, u, v) && t < maxT)
					return true;
			}
		}
		else
		{
			if (l && l->traverseVisible(ray, triangles, maxT))
				return true;
			if (r && r->traverseVisible(ray, triangles, maxT))
				return true;
		}
		return false;
	}
};
