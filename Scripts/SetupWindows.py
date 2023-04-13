import sys
import os
import venv
import subprocess

class ProjectSetupWindows:
    script_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    venv_folder_dir = os.path.join(script_dir, ".venv")
    @classmethod
    def Validate(cls):
        if not cls.ValidatePython():
            return # cannot validate further
        cls.CreateVirtualEnv()
        cls.ActivateVirtualEnv()
        os.chdir("..")
        req_file = os.path.join(os.getcwd(), "requirements.txt")
        with open(f"{req_file}", "r") as file:
            requirements = file.readlines()
        print("\nValidating packages...")
        for requirement in requirements:
            if not requirement.strip() or requirement.strip().startswith('#'):
                continue
            package_name = requirement.split('=')[0].strip()
            if not cls.ValidatePackage(package_name):
                return
        print("\nSetup complete.")
        cls.StartApp()
        
    @classmethod
    def ValidatePython(cls, versionMajor = 3, versionMinor = 3):
        if sys.version is not None:
            print("Python version {0:d}.{1:d}.{2:d} detected.".format( \
                sys.version_info.major, sys.version_info.minor, sys.version_info.micro))
            if sys.version_info.major < versionMajor or (sys.version_info.major == versionMajor and sys.version_info.minor < versionMinor):
                print("Python version too low, expected version {0:d}.{1:d} or higher.".format( \
                    versionMajor, versionMinor))
                return False
            return True
    
    @classmethod
    def CreateVirtualEnv(cls):
        if not os.path.exists(cls.venv_folder_dir):
            os.makedirs(cls.venv_folder_dir)
        print("\nCreating virtual environment...")
        venv.create(
            cls.venv_folder_dir, 
            system_site_packages=False,
            with_pip=True
        )
    
    @classmethod
    def ActivateVirtualEnv(cls):
        if os.name == "nt":
            activate_this = os.path.join(cls.venv_folder_dir, "Scripts", "activate.bat")
        else:
            activate_this = os.path.join(cls.venv_folder_dir, "bin", "activate_this.py")
        print("Activating virtual environment...")
        subprocess.call(f"{activate_this}", shell=True)

    @classmethod
    def ValidatePackage(cls, packageName):
        try:
            output = subprocess.check_output([os.path.abspath("./.venv/Scripts/pip"),"show",packageName], shell=True)
            output_str = output.decode('utf-8')
            if f"Name: {packageName}" in output_str:
                print(f" Package already installed: {packageName}")
                return True
            else:
                cls.InstallPackage(packageName)
        except subprocess.CalledProcessError:
            return cls.InstallPackage(packageName)

    @classmethod
    def InstallPackage(cls, packageName):
        permissionGranted = False
        while not permissionGranted:
            reply = str(input("Would you like to install Python package '{0:s}'? [Y/N]: ".format(packageName))).lower().strip()[:1]
            if reply == "n":
                print("\nWarning: Setup Incomplete, Install packages manually to virtual environment\n")
                return False
            permissionGranted = (reply == "y")
        
        print(f"Installing {packageName} module...")
        subprocess.check_call([os.path.abspath("./.venv/Scripts/pip"), "install", packageName])
        return cls.ValidatePackage(packageName)
    
    @classmethod
    def StartApp(cls):
        permissionGranted = False
        while not permissionGranted:
            reply = str(input("\nRun application now [Y/N]: ")).lower().strip()[:1]
            if reply == "n":
                return
            permissionGranted = (reply == "y")
            
        print("\nStarting application...")
        subprocess.call([os.path.abspath("./.venv/Scripts/streamlit"), "run", "mockfile.py"])

if __name__ == "__main__":
    ProjectSetupWindows.Validate()
