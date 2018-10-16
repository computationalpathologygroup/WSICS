#include <memory>

#include <boost/random/mersenne_twister.hpp>

namespace WSICS::Misc
{
	/// <summary>
	/// A lazy loaded singleton that manages access to a Mersenne Twister 64 bits generator
	/// until the application is exited.
	/// </summary>
	class MT_Singleton
	{
		public:
			MT_Singleton(const MT_Singleton& other)		= delete;
			MT_Singleton(MT_Singleton&& other)			= delete;
			void operator=(const MT_Singleton& other)	= delete;
			void operator=(MT_Singleton&& other)		= delete;

			
			static MT_Singleton& GetInstance(void)
			{
				static MT_Singleton instance;
				return instance;
			}

			static void SetSeed(const size_t seed)
			{
				MT_Singleton& instance(GetInstance());
				instance.m_generator_ = boost::mt19937_64(seed);
			}

			static void SetGenerator(const boost::mt19937_64 generator)
			{
				MT_Singleton& instance(GetInstance());
				instance.m_generator_ = generator;
			}

			static const boost::mt19937_64& GetGenerator()
			{
				MT_Singleton& instance(GetInstance());
				return instance.m_generator_;
			}

		private:
			MT_Singleton()
			{
			}

			boost::mt19937_64 m_generator_;
	};
}