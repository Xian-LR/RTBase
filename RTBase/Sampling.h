#pragma once

#include "Core.h"
#include <random>
#include <algorithm>

class Sampler
{
public:
	virtual float next() = 0;
};

class MTRandom : public Sampler
{
public:
	std::mt19937 generator;
	std::uniform_real_distribution<float> dist;
	MTRandom(unsigned int seed = 1) : dist(0.0f, 1.0f)
	{
		generator.seed(seed);
	}
	float next()
	{
		return dist(generator);
	}
};

// Note all of these distributions assume z-up coordinate system
class SamplingDistributions
{
public:
	static Vec3 uniformSampleHemisphere(float r1, float r2)
	{
		// Add code here
		// Uniformly sample hemisphere using inverse CDF method
		float phi = 2.0f * M_PI * r1;
		float cosTheta = r2;
		float sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);

		// Convert to Cartesian coordinates
		return Vec3(
			std::cos(phi) * sinTheta,
			std::sin(phi) * sinTheta,
			cosTheta
		);
	}
	static float uniformHemispherePDF(const Vec3 wi)
	{
		// Add code here
		return wi.z > 0.0f ? 1.0f / (2.0f * M_PI) : 0.0f;
	}
	static Vec3 cosineSampleHemisphere(float r1, float r2)
	{
		// Add code here
		float a = 2.0f * r1 - 1.0f;
		float b = 2.0f * r2 - 1.0f;

		float r, theta;
		if (a == 0 && b == 0) {
			r = 0;
			theta = 0;
		}
		else if (std::abs(a) > std::abs(b)) {
			r = a;
			theta = (M_PI / 4.0f) * (b / a);
		}
		else {
			r = b;
			theta = (M_PI / 2.0f) - (M_PI / 4.0f) * (a / b);
		}

		float x = r * std::cos(theta);
		float y = r * std::sin(theta);
		float z = std::sqrt(std::max(0.0f, 1.0f - x * x - y * y));
		return Vec3(x, y, z);
	}
	static float cosineHemispherePDF(const Vec3 wi)
	{
		// Add code here
		return wi.z > 0.0f ? wi.z / M_PI : 0.0f;
	}
	static Vec3 uniformSampleSphere(float r1, float r2)
	{
		// Add code here
		// Uniformly sample sphere using inverse CDF method
		float phi = 2.0f * M_PI * r1;
		float cosTheta = 1.0f - 2.0f * r2; // Maps to [-1, 1]
		float sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);

		// Convert to Cartesian coordinates
		return Vec3(
			std::cos(phi) * sinTheta,
			std::sin(phi) * sinTheta,
			cosTheta
		);
	}
	static float uniformSpherePDF(const Vec3& wi)
	{
		// Add code here
		return 1.0f / (4.0f * M_PI);
	}
};