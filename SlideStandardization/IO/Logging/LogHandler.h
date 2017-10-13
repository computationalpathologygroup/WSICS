#ifndef __IO_LOG_HANDLER_H__
#define __IO_LOG_HANDLER_H__

#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "LogLevel.h"

namespace IO::Logging
{
	/// <summary>
	/// A singleton object that places its own origin on the stack. Can be used to log
	/// output towards the command line or to specified files.
	///
	/// This class is thread safe.
	///	</summary>
	/// <remarks>
	/// This class hosts an internal thread that handles the processing of messages, this way the
	/// calling threads will not lose out performance on IO beyond the call and potential mutex
	/// locking.
	///
	///	Three mutexes handle external calls:
	///		m_output_level_access_: Locks access to the output level, reading and writing.
	///		m_command_line_access_: Locks access to the command line messages, reading and writing.
	///		m_file_access_:			Locks access to the file messages, reading and writing.
	///
	/// Two mutexes handle internal calls:
	///		m_in_progress_mutex_:	Locks the destructor from ending the processing until it's finished.
	///		m_notification_mutex_:	Locks the processing thread when there's no messages to process, a new queue addition unlocks it.
	/// </remarks>
    class LogHandler
    {
		public:
			/// <summary>Default constructor, which allows setting of the log level.</summary>
			/// <param name="level">The log level, which defines how much is written to the CMD or a file.</param>
			LogHandler(LogLevel level = LogLevel::SILENT);
			/// <summary>The destructor.</summary>
			~LogHandler(void);

			LogHandler(const LogHandler&)			= delete;
			LogHandler(LogHandler&&)					= delete;
			LogHandler operator=(const LogHandler&) = delete;
			LogHandler operator=(LogHandler&&)		= delete;

			/// <summary>Returns a pointer to the singleton instance.</summary>
			/// <returns>A pointer that references the singleton instance object.</returns>
			static LogHandler* GetInstance(void);
			/// <summary>Registers the object as the global instance to use.</summary>
			/// <param name="log_handler_instance">A pointer towards an instance of the LogHandler.</param>
			static void Register(LogHandler* log_handler_instance);
			
			/// <summary>Initializes the message proccessing thread.</summary>
			void Initialize(void);

			/// <summary>Sets the output level.</summary>
			/// <param name="level">The level of output the singleton should allow.</param>
			void SetOutputLevel(const LogLevel level);
			/// <summary>Returns the current output level.</summary>
			/// <returns>The output level currently in use by the LogHandler.</returns>
			LogLevel GetOutputLevel(void);

			/// <summary>Queue's a message for a log file.</summary>
			/// <param name="message">The message to write to the file.</param>
			/// <param name="file_id">The id of the file to write to.</param>
			/// <param name="level">The level at which this should be written.</param>
			void QueueFileLogging(const std::string message, const size_t file_id, const LogLevel level);
			/// <summary>Queue's a message for the command line.</summary>
			/// <param name="message">The message to write to the command line.</param>
			/// <param name="level">The level at which this should be written.</param>
			void QueueCommandLineLogging(const std::string message, const LogLevel level);

			/// <summary>Opens a file for the LogHandler, allowing messages to be written towards it.</summary>
			/// <param name="filepath">The filepath towards the logfile.</param>
			/// <param name="append">Whether or not the new logs should be appended to potential existing ones.</param>
			/// <returns>The id of the opened file.</returns>
			size_t OpenFile(const std::string filepath, const bool append);
			/// <summary>Closes the specified file. Warning: this will halt any queued messages for the file.</summary>
			/// <param name="file_id">The id of the file to close.</param>
			void CloseFile(const size_t file_id);
			/// <summary>Returns the list of currently open files.</summary>
			/// <returns>A vector holding the list of files currently open.</returns>
			std::vector<std::pair<size_t, std::string>> GetOpenFiles(void);

		private:
			static LogHandler*	m_instance_;

			bool					m_end_operations_;
			LogLevel				m_log_level_;

			std::mutex			m_output_level_access_;
			std::mutex			m_command_line_access_;
			std::mutex			m_file_access_;
			std::mutex			m_in_progress_mutex_;
			std::shared_mutex	m_notification_mutex_;
			std::thread			m_logging_thread_;

			std::unordered_map<size_t, std::pair<std::string, std::ofstream>>	m_open_files_;
			std::queue<std::string>												m_command_line_messages_;
			std::queue<std::pair<size_t, std::string>>							m_file_messages_;

			/// <summary>Starts the thread handling the command line and file writing.</summary>
			void ProcessLogRequests_(void);
			/// <summary>Writes the message to the specified file, removes files that have closed or are causing exceptions.</summary>
			/// <param name="file_id">The id of the file to write to.</param>
			/// <param name="message">The message to write to the file.</param>
			void WriteMessageToFile_(const size_t file_id, const std::string& message);
    };
}
#endif