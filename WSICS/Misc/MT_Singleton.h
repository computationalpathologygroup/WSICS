#include <memory>

#include <boost/random/mersenne_twister.hpp>

namespace WSICS::Misc
{
	/// <summary>
	/// A singleton that manages and allows access to a Mersenne Twister generator until
	/// the pointer holding this object is deleted.
	///
	/// Recommended to retain the instance within the unique_ptr provided by its creation.
	/// </summary>
	class MT_Singleton
	{
		public:
			/// <summary>
			/// Move constructor.
			/// </summary>
			/// <param name="other">Other instance of a MT_Singleton.</param>
			MT_Singleton(MT_Singleton&& other);
			MT_Singleton(const MT_Singleton& other) = delete;

			/// <summary>
			/// Move operator.
			/// </summary>
			/// <param name="other">Other instance of a MT_Singleton.</param>
			/// <returns></returns>
			MT_Singleton& operator=(MT_Singleton&& other);
			MT_Singleton& operator=(const MT_Singleton& other) = delete;
			/// <summary>
			/// Destructor, removes the static pointer towards the instance, and then resumes default deletion.
			/// Assumes actual deletion is based on the stack destructing the unique_pointer owning the actual instance.
			/// </summary>
			~MT_Singleton(void);
			
			/// <summary>
			/// Creates the singleton, without defining a seed for the generator.
			/// </summary>
			/// <returns>A unique pointer that owns the instance.</returns>
			static std::unique_ptr<MT_Singleton> Create(void);
			/// <summary>
			/// Creates the singleton with the specified seed for the generator.
			/// </summary>
			/// <param name="seed">The seed to utilize for the generator.</param>
			/// <returns>A unique pointer that owns the instance.</returns>
			static std::unique_ptr<MT_Singleton> Create(const size_t seed);
			/// <summary>
			/// Returns whether or not the singleton has been initialized.
			/// </summary>
			/// <returns>Whether or not the singleton has been initialized.</returns>
			static bool IsInitialized(void);

			/// <summary>
			/// Returns a reference to the generator.
			/// </summary>
			/// <returns>A reference to the generator.</returns>
			static const boost::mt19937_64& GetGenerator(void);

		private:
			static MT_Singleton* m_instance_;

			boost::mt19937_64 m_generator_;

			/// <summary>
			/// Constructs the singleton without a seed for the generator.
			/// </summary>
			MT_Singleton(void);
			/// <summary>
			/// Constructs the singleton with a seed for the generator.
			/// </summary>
			/// <param name="seed">The seed to utilize for the generator.</param>
			MT_Singleton(const size_t seed);
	};
}