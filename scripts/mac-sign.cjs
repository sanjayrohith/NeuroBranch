const { execFileSync } = require("node:child_process")

module.exports = async function signMacBundle(options) {
  const appPath = options.app

  execFileSync(
    "codesign",
    ["--force", "--deep", "--sign", "-", appPath],
    { stdio: "inherit" },
  )

  execFileSync(
    "codesign",
    ["--verify", "--deep", "--strict", "--verbose=2", appPath],
    { stdio: "inherit" },
  )
}
