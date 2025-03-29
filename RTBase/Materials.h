#pragma once

#include "Core.h"
#include "Imaging.h"
#include "Sampling.h"

#pragma warning( disable : 4244)

class BSDF;

class ShadingData
{
public:
	Vec3 x;
	Vec3 wo;
	Vec3 sNormal;
	Vec3 gNormal;
	float tu;
	float tv;
	Frame frame;
	BSDF* bsdf;
	float t;
	ShadingData() {}
	ShadingData(Vec3 _x, Vec3 n)
	{
		x = _x;
		gNormal = n;
		sNormal = n;
		bsdf = NULL;
	}
};

class ShadingHelper
{
public:
	static float fresnelDielectric(float cosTheta, float iorInt, float iorExt)
	{
		// Add code here
	    // Ensure cosTheta is in the right range
		bool entering = cosTheta > 0.0f;

		// If we're leaving the material, swap the IORs and use the absolute value of cosTheta
		if (!entering) {
			std::swap(iorInt, iorExt);
			cosTheta = std::abs(cosTheta);
		}

		// Compute the relative index of refraction
		float eta = iorExt / iorInt;

		// Compute sin^2(theta_t) using Snell's law
		float sinThetaTSq = eta * eta * (1.0f - cosTheta * cosTheta);

		// Check for total internal reflection
		if (sinThetaTSq >= 1.0f)
			return 1.0f;

		// Calculate cos(theta_t) using the sin^2 value
		float cosThetaT = std::sqrt(1.0f - sinThetaTSq);

		// Calculate Fresnel reflectance using Fresnel equations for dielectrics
		// Rs (perpendicular polarization)
		float Rs = ((iorInt * cosTheta) - (iorExt * cosThetaT)) /
			((iorInt * cosTheta) + (iorExt * cosThetaT));
		Rs = Rs * Rs;

		// Rp (parallel polarization)
		float Rp = ((iorExt * cosTheta) - (iorInt * cosThetaT)) /
			((iorExt * cosTheta) + (iorInt * cosThetaT));
		Rp = Rp * Rp;

		// Average the two polarization states for unpolarized light
		return (Rs + Rp) * 0.5f;
	}
	static Colour fresnelConductor(float cosTheta, Colour ior, Colour k)
	{
		// Add code here
	   // Ensure cosTheta is positive (conductors are not transmissive)
		cosTheta = std::abs(cosTheta);

		// Calculate eta^2 + k^2 for each wavelength
		Colour eta2_plus_k2 = (ior * ior) + (k * k);

		// Calculate 2 * eta * cosTheta for each wavelength
		Colour two_eta_cosTheta = ior * (2.0f * cosTheta);

		// Calculate r_parallel components
		Colour t0 = eta2_plus_k2 * (cosTheta * cosTheta);
		// Correct the operations: addition vs subtraction
		Colour a_plus_b = (t0 + Colour(1.0f, 1.0f, 1.0f)) + two_eta_cosTheta;
		Colour a_minus_b = (t0 + Colour(1.0f, 1.0f, 1.0f)) - two_eta_cosTheta;
		Colour r_parallel = a_minus_b / a_plus_b;

		// Calculate r_perpendicular components
		Colour t1 = eta2_plus_k2 + Colour(cosTheta * cosTheta, cosTheta * cosTheta, cosTheta * cosTheta);
		// Correct the operations: addition vs subtraction
		Colour b_plus_1 = t1 + two_eta_cosTheta;
		Colour b_minus_1 = t1 - two_eta_cosTheta;
		Colour r_perpendicular =  b_minus_1 / b_plus_1 ;

		// Average the two polarization directions for unpolarized light
		return (r_parallel + r_perpendicular) * 0.5f;
	}
	static float lambdaGGX(Vec3 wi, float alpha)
	{
		// Add code here
	    // Check if the direction is valid 
		float cosTheta = wi.z;
		if (cosTheta >= 0.9999f)
			return 0.0f;

		// Calculate tangent^2 of the angle
		float tanThetaSq = (1.0f - cosTheta * cosTheta) / (cosTheta * cosTheta);

		// GGX shadowing function
		return 0.5f * (-1.0f + std::sqrt(1.0f + alpha * alpha * tanThetaSq));
	}

	static float Gggx(Vec3 wi, Vec3 wo, float alpha)
	{
		// Add code here
		// Smith's separable shadowing-masking function with GGX distribution
		return 1.0f / (1.0f + lambdaGGX(wi, alpha) + lambdaGGX(wo, alpha));
	}
	static float Dggx(Vec3 h, float alpha)
	{
		// Add code here
		// Normal distribution function for GGX/Trowbridge-Reitz
		float cosTheta = h.z;
		float cosThetaSq = cosTheta * cosTheta;

		// Check if the half-vector is valid
		if (cosTheta < 1e-6f)
			return 0.0f;

		float tanThetaSq = (1.0f - cosThetaSq) / cosThetaSq;
		float alphaSq = alpha * alpha;

		// Calculation of GGX/Trowbridge-Reitz distribution
		float denom = M_PI * cosThetaSq * cosThetaSq * (alphaSq + tanThetaSq) * (alphaSq + tanThetaSq);
		return alphaSq / denom;
	}
};

class BSDF
{
public:
	Colour emission;
	virtual Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) = 0;
	virtual Colour evaluate(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual bool isPureSpecular() = 0;
	virtual bool isTwoSided() = 0;
	bool isLight()
	{
		return emission.Lum() > 0 ? true : false;
	}
	void addLight(Colour _emission)
	{
		emission = _emission;
	}
	Colour emit(const ShadingData& shadingData, const Vec3& wi)
	{
		return emission;
	}
	virtual float mask(const ShadingData& shadingData) = 0;
};


class DiffuseBSDF : public BSDF
{
public:
	Texture* albedo;
	DiffuseBSDF() = default;
	DiffuseBSDF(Texture* _albedo)
	{
		albedo = _albedo;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Add correct sampling code here
		//Vec3 wi = Vec3(0, 1, 0);
		//pdf = 1.0f;
		//reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		//wi = shadingData.frame.toWorld(wi);
		//return wi;

		// Generate cosine-weighted direction in local space
		Vec3 wiLocal = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());

		// Calculate PDF for this sample (cosine-weighted distribution)
		pdf = wiLocal.z / M_PI;  // cos(theta) / pi

		// Get albedo at the shading point and apply diffuse BRDF (1/π)
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;

		// Transform the local direction to world space
		Vec3 wi = shadingData.frame.toWorld(wiLocal);
		return wi;

	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Add correct PDF code here
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (wiLocal.z <= 0)
			return 0.0f;

		return wiLocal.z / M_PI;
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class MirrorBSDF : public BSDF
{
public:
	Texture* albedo;
	MirrorBSDF() = default;
	MirrorBSDF(Texture* _albedo)
	{
		albedo = _albedo;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with Mirror sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Mirror evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Mirror PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return true;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};


class ConductorBSDF : public BSDF
{
public:
	Texture* albedo;
	Colour eta;
	Colour k;
	float alpha;
	ConductorBSDF() = default;
	ConductorBSDF(Texture* _albedo, Colour _eta, Colour _k, float roughness)
	{
		albedo = _albedo;
		eta = _eta;
		k = _k;
		alpha = 1.62142f * sqrtf(roughness);
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with Conductor sampling code
		// Get outgoing direction in local coordinates
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);

		// Check if direction is valid (above surface)
		if (woLocal.z <= 0)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return Vec3(0.0f, 0.0f, 1.0f);
		}

		// Special case: near-perfect mirror when roughness is very low
		float epsilon = 0.0001f;
		Vec3 wi;

		if (alpha < epsilon)
		{
			// Perfect mirror reflection
			Vec3 wiLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z);
			wi = shadingData.frame.toWorld(wiLocal);
			pdf = 1.0f;

			// Calculate Fresnel reflectance
			float cosTheta = wiLocal.z;
			Colour F = ShadingHelper::fresnelConductor(cosTheta, eta, k);

			// Set color with base albedo and Fresnel factor
			reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) * F;
			return wi;
		}

		// Sample GGX distribution to get microfacet normal
		// Convert uniform random samples to GGX distribution
		float theta = std::atan(alpha * std::sqrt(sampler->next()) / std::sqrt(1 - sampler->next()));
		float phi = 2.0f * M_PI * sampler->next();

		// Convert to Cartesian coordinates
		float sinTheta = std::sin(theta);
		float cosTheta = std::cos(theta);
		Vec3 wh = Vec3(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);


		// Reflect wo around wm to get wi
		Vec3 wiLocal = wh * 2.0f *  woLocal.dot(wh) - woLocal;
		if (wiLocal.z <= 0)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return Vec3(0.0f, 0.0f, 1.0f);
		}

		// Calculate BRDF value
		wi = shadingData.frame.toWorld(wiLocal);
		float D = ShadingHelper::Dggx(wh, alpha);
		pdf = D * wh.z / (4.0f * woLocal.dot(wh));

		// Compute final BRDF 
		Colour F = ShadingHelper::fresnelConductor(woLocal.dot(wh), eta, k);
		float G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);

		Colour baseColor = albedo->sample(shadingData.tu, shadingData.tv);
		reflectedColour = baseColor * F * D * G / (4.0f * woLocal.z);

		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Conductor evaluation code
		// Convert directions to local space
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);

		// Check if both directions are above surface
		if (woLocal.z <= 0 || wiLocal.z <= 0)
			return Colour(0.0f, 0.0f, 0.0f);

		// Compute half vector between wi and wo
		Vec3 wh = (wiLocal + woLocal).normalize();
		if (wh.z <= 0)
			return Colour(0.0f, 0.0f, 0.0f);

		// Perfect mirror case
		float epsilon = 0.0001f;
		if (alpha < epsilon) {
			return Colour(0.0f, 0.0f, 0.0f);
		}

		// Compute Fresnel term
		float cosTheta = std::max(0.0f, wiLocal.dot(wh));
		Colour F = ShadingHelper::fresnelConductor(cosTheta, eta, k);

		// Compute distribution term
		float D = ShadingHelper::Dggx(wh, alpha);

		// Compute shadowing-masking term
		float G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);

		// Microfacet BRDF formula
		Colour baseColor = albedo->sample(shadingData.tu, shadingData.tv);
		Colour brdf = baseColor * F * D * G / (4.0f * wiLocal.z * woLocal.z);

		return brdf;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Conductor PDF
		// Convert directions to local space
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);

		// Check if both directions are above surface
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f)
			return 0.0f;

		// Perfect mirror case
		float epsilon = 0.0001f;
		if (alpha < epsilon) {
			return 0.0f; // Delta distribution
		}

		// Compute half vector
		Vec3 wh = (wiLocal + woLocal).normalize();
		if (wh.z <= 0)
			return 0.0f;

		// Calculate PDF based on GGX distribution
		float D = ShadingHelper::Dggx(wh, alpha);
		return D * wh.z / (4.0f * woLocal.dot(wh));
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class GlassBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	GlassBSDF() = default;
	GlassBSDF(Texture* _albedo, float _intIOR, float _extIOR)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Get the outgoing direction in local coordinates
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);

		// Calculate cosine of incident angle
		float cosTheta = woLocal.z;

		// Calculate Fresnel reflectance
		float Fr = ShadingHelper::fresnelDielectric(cosTheta, intIOR, extIOR);

		// Choose between reflection and refraction based on Fresnel
		bool doReflect = (sampler->next() < Fr);

		Vec3 wiLocal;
		if (doReflect) {
			// Perfect specular reflection: reflect around normal
			wiLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z);
			// PDF is the Fresnel reflectance since we're randomly choosing based on it
			pdf = Fr;
		}
		else {
			// Determine if we're entering or exiting the medium
			bool entering = cosTheta > 0;

			// Adjust IORs and normal direction if needed
			float etaI = entering ? extIOR : intIOR;
			float etaT = entering ? intIOR : extIOR;
			float eta = etaI / etaT;

			// Calculate refraction direction using Snell's law
			float sinThetaISq = std::max(0.0f, 1.0f - cosTheta * cosTheta);
			float sinThetaTSq = eta * eta * sinThetaISq;

			// Check for total internal reflection (should not happen due to Fresnel check)
			if (sinThetaTSq >= 1.0f) {
				// Fall back to reflection
				wiLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z);
				pdf = 1.0f;
			}
			else {
				float cosThetaT = std::sqrt(1.0f - sinThetaTSq);
				// Flip the z component for transmission
				wiLocal = Vec3(-eta * woLocal.x, -eta * woLocal.y,
					entering ? -cosThetaT : cosThetaT);
				pdf = 1.0f - Fr;
			}
		}

		// Transform direction to world space
		Vec3 wi = shadingData.frame.toWorld(wiLocal);

		// Apply the albedo with proper scaling
		// For refraction, we need to account for energy conservation
		Colour baseColor = albedo->sample(shadingData.tu, shadingData.tv);

		if (doReflect) {
			reflectedColour = baseColor;
		}
		else {
			// When refracting, radiance is scaled by (n2/n1)^2
			// This is due to the change in solid angle and energy conservation
			bool entering = cosTheta > 0;
			float etaI = entering ? extIOR : intIOR;
			float etaT = entering ? intIOR : extIOR;
			float scale = (etaT * etaT) / (etaI * etaI);

			reflectedColour = baseColor * scale;
		}

		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Glass evaluation code
		return Colour(0.0f, 0.0f, 0.0f);;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with GlassPDF
		return 0.0f;
	}
	bool isPureSpecular()
	{
		return true;
	}
	bool isTwoSided()
	{
		return false;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class DielectricBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	float alpha;
	DielectricBSDF() = default;
	DielectricBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		alpha = 1.62142f * sqrtf(roughness);
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with Dielectric sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Dielectric evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Dielectric PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return false;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class OrenNayarBSDF : public BSDF
{
public:
	Texture* albedo;
	float sigma;
	OrenNayarBSDF() = default;
	OrenNayarBSDF(Texture* _albedo, float _sigma)
	{
		albedo = _albedo;
		sigma = _sigma;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with OrenNayar sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with OrenNayar evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with OrenNayar PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class PlasticBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	float alpha;
	PlasticBSDF() = default;
	PlasticBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		alpha = 1.62142f * sqrtf(roughness);
	}
	float alphaToPhongExponent()
	{
		return (2.0f / SQ(std::max(alpha, 0.001f))) - 2.0f;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with Plastic sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Plastic evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Plastic PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class LayeredBSDF : public BSDF
{
public:
	BSDF* base;
	Colour sigmaa;
	float thickness;
	float intIOR;
	float extIOR;
	LayeredBSDF() = default;
	LayeredBSDF(BSDF* _base, Colour _sigmaa, float _thickness, float _intIOR, float _extIOR)
	{
		base = _base;
		sigmaa = _sigmaa;
		thickness = _thickness;
		intIOR = _intIOR;
		extIOR = _extIOR;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Add code to include layered sampling
		return base->sample(shadingData, sampler, reflectedColour, pdf);
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Add code for evaluation of layer
		return base->evaluate(shadingData, wi);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Add code to include PDF for sampling layered BSDF
		return base->PDF(shadingData, wi);
	}
	bool isPureSpecular()
	{
		return base->isPureSpecular();
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return base->mask(shadingData);
	}
};