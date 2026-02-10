#ifndef tTabularResults_h
#define tTabularResults_h

#define __MaxColumnNameSize__ 31u
#define __MaxVarChar__ ((2u << 15u) - 3u)


#include <algorithm>
  using std::max;

#include <filesystem>
  using std::filesystem::path;
  using std::filesystem::canonical;
  using std::filesystem::absolute;
  using std::filesystem::exists;
  using std::filesystem::remove;

#include <fstream>
  using std::fstream;
  using std::ofstream;

#include <iostream>
  using std::cout;
  using std::cerr;
  using std::endl;

#include <list>
  using std::list;

#include <map>
  using std::map;

#include <sstream>
  using std::ostringstream;

  #include <cstdint>

#include <string>
  using std::string;
  using std::literals::string_literals::operator""s;

#include <typeinfo>
#include <type_traits>
  using std::is_integral_v;
  using std::is_floating_point_v;
  using std::is_arithmetic_v;

#include <vector>
  using std::vector;

#include "Utility.h"
#include "tException.h"
#include "tStats.h"

enum eColumnType : uint32_t
{
  IsNoType = 0u,
  IsDate,
  IsTime,
  IsTimeStamp,
  IsInteger,
  IsFloat,
  IsChar,
  IsVarChar,
};

inline void
CreateDatabaseSQL(const path &DatabasePath)
{
  path Temp { DatabasePath };

  Temp.replace_extension(R"(.sql)"s);
  fstream ofs(Temp, std::ios_base::out);

  ofs << R"(CREATE DATABASE ')" << DatabasePath.string() << R"(' page_size 8192 user 'SYSDBA' password 'sysdba';)" << endl;

  ofs.close();

  return;
}

inline void
CreateOutputFStream(const path &Path, ofstream &os, std::ios_base::openmode Attributes = std::ios_base::out)
{
  os.open(Path, Attributes);
}

struct tColumnAttributes
{
  inline
  tColumnAttributes(void) : ColumnType(IsNoType), Size(0u), CanBeNull(false) { ColumnIdentifier.clear(); return; }

  inline
  ~tColumnAttributes(void) { return; }

  string      ColumnName;
  string      ColumnIdentifier;
  eColumnType ColumnType;
  uint32_t    Size;
  bool        CanBeNull;

#define tColumnAttributesMemberList ColumnName, ColumnIdentifier, ColumnType, Size, CanBeNull
#define tColumnAttributesDumpList DumpList(tColumnAttributesMemberList)
};

GenDumpStruct(tColumnAttributes)

struct tColumnData
{
  tColumnData(void) = delete;
  inline tColumnData(const string &_ColumnName = R"()", const string &_Value = R"()") : ColumnName(_ColumnName), Value(_Value) { return; }

  string ColumnName;
  string Value;

#define tColumnDataMemberList ColumnName, Value
#define tColumnDataDumpList DumpList(tColumnDataMemberList)
};

GenDumpStruct(tColumnData)

struct tTabularResults
{
  inline
  tTabularResults(void) : InTable(false), InRow(false) { Initialize(); return; }

  virtual inline
  ~tTabularResults(void) { return; }

  inline void
  BeginTable(void)
  {
    if (InTable) Throw(R"(Attempt to BeginTable() while table creation is in progress.)"s);

    // Initialize rows, columns and dictionary.
    Initialize();

   // Show we are creating a table.
    InTable = true;

    return;
  }

  inline void
  BeginRow(void)
  {
    if (!InTable) Throw(R"(Attempt to invoke BeginRow(), but without prior call to BeginTable().)"s);

    ColumnData.push_back({ });

    InRow = true;

    return;
  }

private:
  inline void
  AddColumn(const string &DataColumnName, const string &DataValue, eColumnType DataType, uint32_t DataSize)
  {
    if (!InTable) Throw(R"(Attempt to add a column without prior call to BeginTable().)"s);
    if (!InRow) Throw(R"(Attempt to add a column without prior call to BeginRow().)"s);

    auto TableColumnName { DataColumnName };
    auto TableColumnIdentifier { ValidateIdentifier(TableColumnName) };

    ExposeStruct(tColumnAttributes, ColumnsAttributes[TableColumnIdentifier]);

    // Handle first use.
    if (ColumnType == IsNoType)
    {
      // Set type.
      ColumnType = DataType;

      // Set size.
      Size = DataSize;

      // Set column name.
      ColumnName = TableColumnName;
      ColumnIdentifier = TableColumnIdentifier;
      Columns.push_back(TableColumnIdentifier);
    }
    else
    {
      // Get new size.
      auto NewSize = max(Size, DataSize);

      // Handle integer to float promotion..
      if ((ColumnType == IsInteger) && (DataType == IsFloat))
      {
        // Handle promoting integer to float.
        ColumnType = IsFloat;
      }

      //  If column type is already VARCHAR and data type is CHAR, force data type to VARCHAR.
      if ((ColumnType == IsVarChar) && (DataType == IsChar))
      {
        DataType = IsVarChar;
      }

      // If data type is CHAR and size changes, promote to VARCHAR.
      if ((DataSize != Size) && (ColumnType == IsChar))
      {
        DataType = IsVarChar;
        ColumnType = IsVarChar;
      }

      // Check for illegal promotion.
      if (ColumnType != DataType)
      {
        Throw(R"(Illegal type change.)"s);
      }

      // Set new size.
      Size = NewSize;
    }

    // Remember the data.
    ColumnData.back()[TableColumnIdentifier] = DataValue;

    return;
  }

public:
  inline void
  AddColumnDate(const string &ColumnName, const string &Value)
  {
    AddColumn(ColumnName, Trim(Value), IsDate, 1u);

    return;
  }

  inline void
  AddColumnTime(const string &ColumnName, const string &Value)
  {
    AddColumn(ColumnName, Trim(Value), IsTime, 1u);

    return;
  }

  inline void
  AddColumnTimestamp(const string &ColumnName, const string &Value)
  {
    AddColumn(ColumnName, Trim(Value), IsTimeStamp, 2u);

    return;
  }

  inline void
  AddColumn(const string &ColumnName, const char Value)
  {
    string Temp { Value };
    AddColumn(ColumnName, Temp,  IsChar, 1u);

    return;
  }

  inline void
  AddColumn(const string &ColumnName, const char *Value)
  {
    string Temp { Value };
    AddColumn(ColumnName, Temp, IsChar, Temp.length());

    return;
  }

  inline void
  AddColumn(const string &ColumnName, const string &Value)
  {
    AddColumn(ColumnName, Value, IsChar, Value.length());

    return;
  }

  template<typename T>
  inline void
  AddColumn(const string &ColumnName, const T &Value, const string &Fmt = R"(%20.16e)"s)
  {
    if (is_integral_v<T>)
    {
      // Handle integer type.
      AddColumn(ColumnName, ToString(Value), IsInteger, sizeof(T));
    }
    else if (is_floating_point_v<T>)
    {
      // Format to string.
      auto ValueString { Trim(Format(Fmt, Value)) };

      AddColumn(ColumnName, ValueString, IsFloat, sizeof(T));
    }
    else
    {
      // If one of the specializations does not process this type, then this type is not a illegal SQL type.
      Throw(R"(Illegal SQL data type.)"s);
    }

    return;
  }

  template<class T=double>
  inline void AddColumn(const string &Key, const tStats<T> &Value)
  {
    AddColumn(Key + R"(_N)"s, (uint64_t) Value.N());
    AddColumn(Key + R"(_Avg)"s, (Value.N() >= 1.0) ? Value.Avg()     : 0.0);
    AddColumn(Key + R"(_Sum)"s, (Value.N() >= 1.0) ? Value.SumX      : 0.0);
    AddColumn(Key + R"(_Min)"s, (Value.N() >= 1.0) ? Value.Minimum() : 0.0);
    AddColumn(Key + R"(_Max)"s, (Value.N() >= 1.0) ? Value.Maximum() : 0.0);
    AddColumn(Key + R"(_Var)"s, (Value.N() >= 2.0) ? Value.SampVar() : 0.0);
    AddColumn(Key + R"(_Std)"s, (Value.N() >= 2.0) ? Value.SampStd() : 0.0);

    return;
  }

  inline void
  EndRow(void)
  {
    InRow = false;

    return;
  }

  inline void
  EndTable(void)
  {
    InTable = false;

    return;
  }

  inline void
  AdjustColumnType(const string &ColumnName, const eColumnType ColumnType, const uint32_t Size)
  {
    auto ColumnIdentifier { ValidateIdentifier(ColumnName) };

    if (ColumnsAttributes.count(ColumnIdentifier) == 0u) Throw(Format(R"(Column "%s" does not exist.)"s, ColumnName.c_str()));

    ColumnsAttributes[ColumnIdentifier].ColumnType = ColumnType;
    ColumnsAttributes[ColumnIdentifier].Size = Size;

    return;
  }

  inline void
  AdjustColumnCanBeNull(const string &ColumnName, bool CanBeNull)
  {
    auto ColumnIdentifier { ValidateIdentifier(ColumnName) };

    if (ColumnsAttributes.count(ColumnIdentifier) == 0u) Throw(Format(R"(Column "%s" does not exist.)"s, ColumnName.c_str()));

    ColumnsAttributes[ColumnIdentifier].CanBeNull = CanBeNull;

    return;
  }

  inline void
  CreateCSV(ostream &os)
  {
    // Create column header
    const char *Sep { R"(")" };
    for (const auto &Column : Columns)
    {
      ExposeStruct(tColumnAttributes, ColumnsAttributes[Column]);
      os << Sep << ColumnName << R"(")";
      Sep = R"(,")";
    }
    os << endl;

    // Create body of table.

    // For each row in the order entered . . .
    for (auto &Row : ColumnData)
    {

      Sep = R"()";

      // For each column in Row insert column values . . .
      for (const auto &Col : Columns)
      {
        if (Row.count(Col))
        {
          // Get attributes for column.
          ExposeStructConst(tColumnAttributes, ColumnsAttributes[Col]);

          // Figure out which values must be quoted.
          string Quote { R"()" };
          switch (ColumnType)
          {
            case IsInteger:
            case IsFloat:
              Quote = R"()";
              break;

            case IsDate:
            case IsTime:
            case IsTimeStamp:
            case IsChar:
            case IsVarChar:
              Quote = R"(")";
              break;

            case IsNoType:
            default:
              Throw(R"(No data type available while adding row.)"s);
          };

          os << Sep << Quote << Row[Col] << Quote;
        }
        else
        {
          os << Sep;
        }
        Sep = R"(,)";
      }
      os << endl;
    }

    return;
  }

  inline void
  CreateTableSQL(ostream &os, const string &TableName)
  {
    string Sep { R"()" };

    ValidateIdentifier(TableName);

    os << R"(DROP TABLE )" + TableName << R"(;)"s << endl << endl;;

    os << R"(CREATE TABLE )" + TableName << endl;

    os << R"(()" << endl;

    // For each column in the table in the order it appeared . . .
    for (const auto &Column : Columns)
    {
      ExposeStruct(tColumnAttributes, ColumnsAttributes[Column]);

      if (Sep.length())
      {
        os << Sep << endl;
      }

      os << R"(  )" << ColumnName << R"( )" << GetSQLType(ColumnType, Size);

      if (!CanBeNull)
      {
        os << R"( NOT NULL)";
      }

      Sep = R"(,)";
    }

    os << endl << R"();)" << endl;

    return;
  }

  inline void
  CreateInsertSQL(ostream &os, const string &TableName, const vector<string> &AfterEachInsert = {})
  {
    // For each row in the order entered . . .
    for (auto &Row : ColumnData)
    {
      // Row preamble . . .
      os << R"(INSERT INTO )" + ValidateIdentifier(TableName) + R"( ( )";

      // Columns . . .
      string Sep { R"()"s };

      // For each column in Row insert column names . . .
      for (const auto &Col : Columns)
      {
        // Get attributes for column.
        os << Sep << ColumnsAttributes[Col].ColumnName;
        Sep = R"(,)"s;
      }
      os << R"( ))";
      os << endl;
      os << R"(VALUES ( )";

      Sep = R"()"s;

      // For each column in Row insert column values . . .
      for (const auto &Col : Columns)
      {
        if (Row.count(Col))
        {
          // Get attributes for column.
          ExposeStructConst(tColumnAttributes, ColumnsAttributes[Col]);

          // Figure out which values must be quoted.
          string Quote { R"()"s };
          switch (ColumnType)
          {
            case IsInteger:
            case IsFloat:
              Quote = R"()";
              break;

            case IsDate:
            case IsTime:
            case IsTimeStamp:
            case IsChar:
            case IsVarChar:
              Quote = R"(')";
              break;

            case IsNoType:
            default:
              Throw(R"(No data type available while generating INSERT SQL)"s);
          };

          os << Sep << Quote << Row[Col] << Quote;
        }
        else
        {
          os << Sep;
        }
        Sep = R"(,)"s;
      }

      os << R"( );)" << endl;
      for (const auto &AfterInsertLine : AfterEachInsert)
      {
        os << AfterInsertLine << endl;
      }
    }

    return;
  }

private:
  inline string
  GetSQLType(const eColumnType ColumnType, const uint32_t Size) const
  {
    if (Size == 0u)
    {
      Throw(R"(Size of SQL data type cannot be zero.)"s);
    }

    switch (ColumnType)
    {
      case IsNoType:
        Throw(R"(No type set.)"s);

      case IsDate:
        return R"(DATE)"s;

      case IsTime:
        return R"(TIME)"s;

      case IsTimeStamp:
        return R"(TIMESTAMP)"s;

      case IsInteger:
        if (Size <= 2u)
        {
          return R"(SMALLINT)"s;
        }
        else if (Size <= 4u)
        {
          return R"(INTEGER)"s;
        }
        else
        {
          return R"(BIGINT)"s;
        }

      case IsFloat:
        if (Size <= 4u)
        {
          return R"(FLOAT)"s;
        }
        else
        {
          return R"(DOUBLE PRECISION)"s;
        }

      case IsChar:
          return R"(CHAR()"s + ToString(Size) + R"())"s;

      case IsVarChar:
        return R"(VARCHAR()"s + ToString(Size) + R"())"s;

      default:
        Throw(R"(Invalid type.)"s);
    }
  }

  inline void
  Initialize(void)
  {
    Columns.clear();
    ColumnsAttributes.clear();
    ColumnData.clear();

    return;
  }

  inline string
  ValidateIdentifier(const string &Identifier) const
  {
    if (Identifier.length() > __MaxColumnNameSize__)
    {
      Throw(Format(R"(Column/Table name "%s" exceeds %u characters.)"s, Identifier.c_str(), __MaxColumnNameSize__));
    }

    return ToUpper(Identifier);
  }

  inline tColumnAttributes *
  ColumnNameExists(const string &ColumnName)
  {
    return ColumnsAttributes.count(ColumnName) ? &ColumnsAttributes[ColumnName] : nullptr;
  }

public:
  list<string> Columns;
  map<string, tColumnAttributes> ColumnsAttributes;
  vector<map<string, string>> ColumnData;
  bool InTable;
  bool InRow;
};

#endif
