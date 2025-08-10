// JSON Renderer utility for formatting and displaying JSON data with syntax highlighting

/**
 * Renders JSON value with appropriate styling and formatting
 * @param {*} value - The value to render
 * @param {string} key - The key name (optional)
 * @returns {JSX.Element} - Formatted JSX element
 */
export const renderJsonValue = (value, key = '') => {
  if (typeof value === 'string') {
    return <span className="text-emerald-400">"{value}"</span>;
  }
  
  if (typeof value === 'number') {
    return <span className="text-cyan-400">{value}</span>;
  }
  
  if (typeof value === 'boolean') {
    return <span className="text-purple-400">{value.toString()}</span>;
  }
  
  if (Array.isArray(value)) {
    return renderArray(value);
  }
  
  if (typeof value === 'object' && value !== null) {
    return renderObject(value);
  }
  
  return <span className="text-slate-300">{String(value)}</span>;
};

/**
 * Renders an array with proper formatting
 * @param {Array} array - Array to render
 * @returns {JSX.Element} - Formatted array JSX
 */
const renderArray = (array) => {
  return (
    <div>
      <span className="text-slate-300">[</span>
      <div className="ml-4">
        {array.map((item, index) => (
          <div key={index} className="flex">
            {renderJsonValue(item)}
            {index < array.length - 1 && <span className="text-slate-300">,</span>}
          </div>
        ))}
      </div>
      <span className="text-slate-300">]</span>
    </div>
  );
};

/**
 * Renders an object with proper formatting
 * @param {Object} obj - Object to render
 * @returns {JSX.Element} - Formatted object JSX
 */
const renderObject = (obj) => {
  const entries = Object.entries(obj);
  
  return (
    <div>
      <span className="text-slate-300">{'{'}</span>
      <div className="ml-4">
        {entries.map(([k, v], index) => (
          <div key={k} className="flex flex-wrap items-start gap-1">
            <span className="text-amber-400">"{k}"</span>
            <span className="text-slate-300">:</span>
            <div className="flex-1">
              {renderJsonValue(v, k)}
            </div>
            {index < entries.length - 1 && <span className="text-slate-300">,</span>}
          </div>
        ))}
      </div>
      <span className="text-slate-300">{'}'}</span>
    </div>
  );
};

/**
 * Converts JSON to formatted string with indentation
 * @param {*} obj - Object to stringify
 * @param {number} indent - Indentation level
 * @returns {string} - Formatted JSON string
 */
export const formatJsonString = (obj, indent = 2) => {
  try {
    return JSON.stringify(obj, null, indent);
  } catch (error) {
    console.error('Error formatting JSON:', error);
    return 'Error formatting response data';
  }
};

/**
 * Validates if a string is valid JSON
 * @param {string} str - String to validate
 * @returns {boolean} - True if valid JSON
 */
export const isValidJson = (str) => {
  try {
    JSON.parse(str);
    return true;
  } catch {
    return false;
  }
};

/**
 * Safely parses JSON string
 * @param {string} str - JSON string to parse
 * @param {*} defaultValue - Default value if parsing fails
 * @returns {*} - Parsed object or default value
 */
export const safeJsonParse = (str, defaultValue = null) => {
  try {
    return JSON.parse(str);
  } catch (error) {
    console.warn('Failed to parse JSON:', error);
    return defaultValue;
  }
};