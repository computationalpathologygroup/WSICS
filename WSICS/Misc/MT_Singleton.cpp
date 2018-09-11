#include "MT_Singleton.h"

#include <stdexcept>

namespace WSICS::Misc
{
	MT_Singleton::MT_Singleton(void) : m_generator_()
	{
	}

	MT_Singleton::MT_Singleton(const size_t seed) : m_generator_(seed)
	{
	}

	MT_Singleton::MT_Singleton(MT_Singleton&& other) : m_generator_(std::move(other.m_generator_))
	{
	}

	MT_Singleton& MT_Singleton::operator=(MT_Singleton&& other)
	{
		this->m_generator_ = std::move(other.m_generator_);
		return *this;
	}

	MT_Singleton::~MT_Singleton(void)
	{
		m_instance_ = nullptr;
	}

	std::unique_ptr<MT_Singleton> MT_Singleton::Create(void)
	{
		if (IsInitialized())
		{
			std::runtime_error("An instance of the MT Singleton already exists.");
		}

		m_instance_ = new MT_Singleton();
		return std::unique_ptr<MT_Singleton>(m_instance_);
	}

	std::unique_ptr<MT_Singleton> MT_Singleton::Create(const size_t seed)
	{
		if (IsInitialized())
		{
			std::runtime_error("An instance of the MT Singleton already exists.");
		}

		m_instance_ = new MT_Singleton(seed);
		return std::unique_ptr<MT_Singleton>(m_instance_);
	}

	bool MT_Singleton::IsInitialized(void)
	{
		return m_instance_;
	}

	const boost::mt19937_64& MT_Singleton::GetGenerator(void)
	{
		if (!IsInitialized())
		{
			std::runtime_error("No instance of the MT_Singleton has been initialized.");
		}

		return m_instance_->m_generator_;
	}
}