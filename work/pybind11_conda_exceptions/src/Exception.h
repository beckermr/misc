#ifndef EXCEPTION_H
#define EXCEPTION_H

#include <exception>
#include <ostream>
#include <string>

#define LSST_EXPORT

namespace exceptions {

class LSST_EXPORT LSSTException : public std::exception {
  public:
      LSSTException(std::string const& message);
      virtual ~LSSTException(void) noexcept;
      virtual char const* what(void) const noexcept;
      virtual LSSTException* clone(void) const;

  private:
      std::string _message;
};


class LSST_EXPORT CustomError : public LSSTException {
  public:
      CustomError(std::string const& message) : LSSTException(message){};
      virtual exceptions::LSSTException* clone(void) const { return new CustomError(*this); };
};

}

#endif
