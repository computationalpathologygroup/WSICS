#include "LogHandler.h"

#include <utility>

namespace WSICS::IO::Logging
{
	// Initialises the static variable.
	LogHandler* LogHandler::m_instance_ = nullptr;

	LogHandler::LogHandler(LogLevel level) : m_log_level_(level)
	{
	}

	LogHandler::~LogHandler(void)
	{
		m_end_operations_ = true;

		if (m_command_line_messages_.empty() && m_file_messages_.empty())
		{
			m_notification_mutex_.unlock();
		}

		m_in_progress_mutex_.lock();
		m_in_progress_mutex_.unlock();
		m_logging_thread_.join();

		for (std::pair<const size_t, OpenedFile>& open_file : m_open_files_)
		{
			open_file.second.filestream.close();
		}
	}

	LogHandler* LogHandler::GetInstance(void)
	{
		return LogHandler::m_instance_;
	}

	void LogHandler::Register(LogHandler* log_handler_instance)
	{
		LogHandler::m_instance_ = log_handler_instance;
	}

	void LogHandler::Initialize(void)
	{
		m_logging_thread_ = std::thread(&LogHandler::ProcessLogRequests_, this);
	}

	void LogHandler::SetOutputLevel(const LogLevel level)
	{
		m_output_level_access_.lock();
		m_log_level_ = level;
		m_output_level_access_.unlock();
	}

	LogLevel LogHandler::GetOutputLevel(void)
	{
		return m_log_level_;
	}

	void LogHandler::QueueFileLogging(const std::string message, const size_t file_id, const LogLevel level)
	{
		m_output_level_access_.lock();
		if (level <= m_log_level_)
		{
			m_file_access_.lock();
			m_file_messages_.push({file_id, message});
			if (!m_notification_mutex_.try_lock_shared())
			{
				m_notification_mutex_.unlock_shared();
			}
			m_file_access_.unlock();
		}
		m_output_level_access_.unlock();
	}

	void LogHandler::QueueCommandLineLogging(const std::string message, const LogLevel level)
	{
		m_output_level_access_.lock();
		if (level <= m_log_level_)
		{
			m_command_line_access_.lock();
			m_command_line_messages_.push(message);
			if (!m_notification_mutex_.try_lock_shared())
			{
				m_notification_mutex_.unlock_shared();
			}
			m_command_line_access_.unlock();
		}
		m_output_level_access_.unlock();
	}

	size_t LogHandler::OpenFile(const std::string filepath, const bool append)
	{
		// Sets the flags to open the file with.
		std::ios_base::openmode flags;
		if (append)
		{
			flags = std::ofstream::out | std::ofstream::app;
		}
		else
		{
			flags = std::ofstream::out;
		}

		// Checks if the stream can be openend and then inserts the file into the list.
		std::ofstream stream(filepath, flags);
		if (stream.is_open())
		{
			stream.close();
			m_file_access_.lock();

			size_t file_index = 1;
			if (!m_open_files_.empty())
			{
				auto file_iterator = m_open_files_.end();
				file_iterator.operator++();
				file_index = file_iterator->first + 1;
			}

			std::pair<size_t, OpenedFile> insertion_pair = { file_index, OpenedFile{ filepath, std::ofstream(filepath, flags) } };
			m_open_files_.insert(std::move(insertion_pair));

			m_file_access_.unlock();
			return file_index;
		}
		else
		{
			std::runtime_error("Couldn't open the file: " + filepath);
		}
		
		// Unreachable, but prevents a warning.
		return (size_t)-1;
	}

	void LogHandler::CloseFile(const size_t file_id)
	{
		m_file_access_.lock();
		if (m_open_files_.find(file_id) != m_open_files_.end())
		{
			m_open_files_.erase(file_id);
		}
		m_file_access_.unlock();
	}

	std::vector<std::pair<size_t, std::string>> LogHandler::GetOpenFiles(void)
	{
		std::vector<std::pair<size_t, std::string>> files;
		for (const std::pair<const size_t, OpenedFile>& file : m_open_files_)
		{
			files.push_back({ file.first, file.second.filepath });
		}
		return files;
	}

	void LogHandler::ProcessLogRequests_(void)
	{
		m_in_progress_mutex_.lock();

		// Loops until the instance is destroyed.
		m_end_operations_ = false;
		while (!m_end_operations_)
		{
			m_notification_mutex_.lock_shared();

			// Loops through all the command line messages, outputting them to the console.
			while (!m_command_line_messages_.empty())
			{
				m_command_line_access_.lock();
				std::string command_line_message(m_command_line_messages_.front());
				m_command_line_messages_.pop();
				m_command_line_access_.unlock();
				std::cout << command_line_message << std::endl;
			}

			// Loops through all the file messages, writing them to the files.
			while (!m_file_messages_.empty())
			{
				m_file_access_.lock();
				size_t		file_id(m_file_messages_.front().first);
				std::string file_message(m_file_messages_.front().second);
				m_file_messages_.pop();
				m_file_access_.unlock();
				WriteMessageToFile_(file_id, file_message);
			}

			// Locks if there's no more available messages to process.
			if (m_command_line_messages_.empty() && m_file_messages_.empty())
			{
				m_notification_mutex_.lock_shared();
			}
			else
			{
				m_notification_mutex_.unlock_shared();
			}
		}
		
		m_in_progress_mutex_.unlock();
	}

	void LogHandler::WriteMessageToFile_(const size_t file_id, const std::string& message)
	{
		auto open_file_iterator = m_open_files_.find(file_id);

		if (open_file_iterator != m_open_files_.end())
		{
			std::ofstream& stream = open_file_iterator->second.filestream;

			if (stream.is_open())
			{
				try
				{
					stream.write((message + "\n").c_str(), message.size() + 1);
				}
				catch (...)
				{
					CloseFile(file_id);
				}
			}
		}
	}
}