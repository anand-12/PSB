#!/usr/bin/env python3
# ==============================================================================
# Cloud TPU - Cross-Platform General Diagnostic & Quota Verification Utility
# ==============================================================================
# Run this script in Google Cloud Shell or in any terminal authenticated with gcloud.
# Compatible with Linux, macOS, and Windows. Requires only Python 3.6+.
# ==============================================================================

import os
import sys
import json
import subprocess
import shutil
import re
import concurrent.futures

class Tee(object):
    def __init__(self, filename):
        self.filename = filename
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def __enter__(self):
        self.file = open(self.filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        try:
            self.file.close()
        except Exception:
            pass

    def write(self, data):
        clean_data = self.ansi_escape.sub('', data)
        self.file.write(clean_data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def isatty(self):
        return self.stdout.isatty()

# Terminal colors
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m' # No Color

# Badges
PASS = f"[{GREEN}PASS{NC}]"
FAIL = f"[{RED}FAIL{NC}]"
WARN = f"[{YELLOW}WARN{NC}]"
INFO = f"[{BLUE}INFO{NC}]"

# Check if terminal supports colors (handles windows and pipe redirections)
if sys.platform == 'win32' or not sys.stdout.isatty():
    # Enable colored terminal support on modern Windows 10+ consoles
    if sys.platform == 'win32':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            # Strip colors if console setup fails
            RED = GREEN = YELLOW = BLUE = NC = ""
            PASS = "[PASS]"
            FAIL = "[FAIL]"
            WARN = "[WARN]"
            INFO = "[INFO]"
    else:
        RED = GREEN = YELLOW = BLUE = NC = ""
        PASS = "[PASS]"
        FAIL = "[FAIL]"
        WARN = "[WARN]"
        INFO = "[INFO]"

RECOMMENDED_ZONES = ["us-central1-a", "us-east5-a", "us-east5-b", "europe-west4-a", "southamerica-west1-a", "us-west4-a", "europe-west4-b"]

# All zones we want to scan for dangling resources
SCAN_ZONES = [
    "us-east5-a", "us-east5-b", "us-east5-c",
    "us-west4-a", "us-west4-b", "us-west4-c",
    "us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f",
    "southamerica-west1-a", "southamerica-west1-b", "southamerica-west1-c",
    "europe-west4-a", "europe-west4-b", "europe-west4-c"
]

GCLOUD_EXEC = shutil.which("gcloud")

def run_gcloud_cmd(args, timeout=10):
    """Safely executes a gcloud command and returns output or an error string."""
    if not GCLOUD_EXEC:
        return "ERROR: gcloud SDK not found on system PATH."
    
    cmd = [GCLOUD_EXEC] + args
    kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "check": True,
        "timeout": timeout
    }
    
    if sys.platform == 'win32':
        # Suppress flashing command prompt windows on older Windows systems
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

    try:
        result = subprocess.run(cmd, **kwargs)
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out."
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.stderr.strip()}"
    except FileNotFoundError:
        return "ERROR: gcloud SDK not found on system PATH."

def run_gcloud_json(args, timeout=10):
    """Safely executes a gcloud command with JSON formatting and parses the JSON output."""
    res = run_gcloud_cmd(args + ["--format=json"], timeout=timeout)
    if "ERROR" in res:
        return None, res
    try:
        return json.loads(res), None
    except json.JSONDecodeError:
        return None, f"ERROR: Invalid JSON response. Raw output: {res[:100]}"

def main():
    print(f"{BLUE}========================================================")
    print(f"   Cloud TPU - Cross-Platform General Diagnostic Utility")
    print(f"========================================================{NC}\n")

    # Global metrics for summary
    sdk_version = "Unknown"
    api_tpu_ok = False
    api_compute_ok = False
    billing_state = "UNKNOWN"
    billing_id = "Unknown"
    alpha_ok = False
    default_vpc_ok = False
    ce_sa_ok = False
    tpu_sa_ok = False
    global_accel_ok = False
    has_zero_quota = False
    locations_ok = False
    visible_zones = set()

    # 1. Check gcloud Authentication
    print("🔍 [1/12] Checking gcloud authentication...")
    account = run_gcloud_cmd(["config", "get-value", "account"])
    if not account or "ERROR" in account or account == "None":
        print(f"  {FAIL} No active gcloud account found. Please run 'gcloud auth login' first.")
        sys.exit(1)
    print(f"  {PASS} Authenticated as: {GREEN}{account}{NC}")

    # 2. Check Active Project
    print("\n🔍 [2/12] Retrieving project metadata...")
    project_id = run_gcloud_cmd(["config", "get-value", "project"])
    if not project_id or "ERROR" in project_id or project_id == "None":
        print(f"  {FAIL} No active Google Cloud project set. Run 'gcloud config set project <PROJECT_ID>'")
        sys.exit(1)

    project_data, err = run_gcloud_json(["projects", "describe", project_id])
    if err:
        print(f"  {FAIL} Failed to query project metadata. Verify your permissions on project: {project_id}")
        print(f"         Raw details: {err}")
        sys.exit(1)

    project_number = project_data.get("projectNumber", "Unknown")
    print(f"  ✅ Project ID:     {GREEN}{project_id}{NC}")
    print(f"  ✅ Project Number: {GREEN}{project_number}{NC}")

    # 3. Check Billing Status
    print("\n🔍 [3/12] Checking billing status...")
    billing_data, err = run_gcloud_json(["beta", "billing", "projects", "describe", project_id])
    if err:
        print(f"  {INFO} Unable to query billing status (permission missing on billing account).")
        print(f"         Raw details: {err}")
        print("         Note: If you have active billing linked, this is safe to ignore.")
        billing_state = "WARN"
    else:
        enabled = billing_data.get("billingEnabled", False)
        account_name = billing_data.get("billingAccountName", "")
        billing_id = account_name.split("/")[-1] if account_name else "None"
        
        if enabled:
            print(f"  {PASS} Billing is ENABLED for this project (Billing Account ID: {GREEN}{billing_id}{NC}).")
            billing_state = "OK"
        else:
            print(f"  {FAIL} Billing is DISABLED. Cloud TPUs require an active billing account.")
            billing_state = "FAIL"

    # 4. Check gcloud components
    print("\n🔍 [4/12] Checking gcloud components...")
    version_data, err = run_gcloud_json(["version"])
    if version_data:
        sdk_version = version_data.get("Google Cloud SDK", "Unknown")
        print(f"  ✅ Google Cloud SDK Version: {GREEN}{sdk_version}{NC}")
    else:
        print(f"  {WARN} Unable to query Google Cloud SDK version: {err}")

    comp_data, err = run_gcloud_json(["components", "list", "--filter=id:alpha"])
    if comp_data and comp_data[0].get("state", {}).get("name") == "Installed":
        print(f"  {PASS} gcloud 'alpha' components are installed (required for queued resources).")
        alpha_ok = True
    else:
        print(f"  {WARN} gcloud 'alpha' components are NOT installed.")
        print("         To install: gcloud components install alpha --quiet")

    # 5. Check Network Configuration (VPC)
    print("\n🔍 [5/12] Checking network configuration (VPC)...")
    vpc_data, err = run_gcloud_json(["compute", "networks", "list", "--filter=name:default"])
    if vpc_data and vpc_data[0].get("name") == "default":
        print(f"  {PASS} 'default' VPC network exists.")
        default_vpc_ok = True
    else:
        print(f"  {WARN} 'default' VPC network is missing.")
        print(f"         Details: {err if err else 'default VPC missing'}")
        print("         Note: You must specify your custom network/subnet when creating TPU VMs.")

    # 6. Check Organization Policies
    print("\n🔍 [6/12] Checking organizational policies...")
    policy_data, err = run_gcloud_json(["resource-manager", "org-policies", "describe", "constraints/compute.vmExternalIpAccess", f"--project={project_id}"])
    if policy_data and policy_data.get("constraint") == "constraints/compute.vmExternalIpAccess" and ("listPolicy" in policy_data or "booleanPolicy" in policy_data or "rules" in policy_data):
        print(f"  {WARN} Org policy 'compute.vmExternalIpAccess' is active.")
        print("         Note: This restricts public IP allocation and can cause TPU VM creations to fail.")
        print("               Ensure your TPU command is configured for private network setups if needed.")
    else:
        print(f"  {PASS} No restricting external IP policies detected.")

    # 7. Check Required APIs
    print("\n🔍 [7/12] Checking required APIs...")
    for api in ["tpu.googleapis.com", "compute.googleapis.com"]:
        api_data, err = run_gcloud_json(["services", "list", "--enabled", f"--filter=config.name:{api}"])
        if api_data and api_data[0].get("state") == "ENABLED":
            print(f"  {PASS} {api} is ENABLED")
            if api == "tpu.googleapis.com":
                api_tpu_ok = True
            else:
                api_compute_ok = True
        else:
            print(f"  {FAIL} {api} is DISABLED.")
            print(f"         To enable: gcloud services enable {api}")

    # 8. Check Service Account Roles
    print("\n🔍 [8/12] Checking TPU and Compute service accounts...")
    ce_sa = f"{project_number}-compute@developer.gserviceaccount.com"
    tpu_sa = f"service-{project_number}@cloud-tpu.iam.gserviceaccount.com"

    iam_data, err = run_gcloud_json(["projects", "get-iam-policy", project_id])
    if err:
        print(f"  {WARN} Unable to query project IAM roles (permission missing).")
        print(f"         Raw details: {err}")
    else:
        bindings = iam_data.get("bindings", [])
        
        ce_has_perms = False
        tpu_has_perms = False
        
        for binding in bindings:
            members = binding.get("members", [])
            role = binding.get("role", "")
            
            if any(ce_sa in m for m in members):
                role_lower = role.lower()
                if "storage" in role_lower or "editor" in role_lower or "owner" in role_lower:
                    ce_has_perms = True
            
            if any(tpu_sa in m for m in members):
                tpu_has_perms = True
                
        if ce_has_perms:
            print(f"  {PASS} Default Compute Engine SA has storage permissions.")
            ce_sa_ok = True
        else:
            print(f"  {WARN} Default Compute Engine SA permissions are restricted.")
            print("         Note: You may get 403 Forbidden when writing GCS checkpoints from your TPU VM.")
            
        if tpu_has_perms:
            print(f"  {PASS} TPU Service Agent registered.")
            tpu_sa_ok = True
        else:
            print(f"  {WARN} TPU Service Agent role missing.")
            print("         To create: gcloud beta services identity create --service tpu.googleapis.com")

    # 9. Check Global Accelerator Quota
    print("\n🔍 [9/12] Checking Global Accelerator Quota (GPUS_ALL_REGIONS)...")
    quota_data, err = run_gcloud_json(["compute", "project-info", "describe"], timeout=15)
    if err:
        print(f"  {WARN} Unable to query global project quota automatically: {err}")
    else:
        quotas = quota_data.get("quotas", [])
        gpu_quota = [q for q in quotas if q.get("metric") == "GPUS_ALL_REGIONS"]
        if gpu_quota:
            q = gpu_quota[0]
            limit = q.get("limit", 0.0)
            usage = q.get("usage", 0.0)
            print("  ✅ Metric: %-20s | Limit: %-5s | Usage: %-5s" % (q.get("metric"), limit, usage))
            if limit == 0.0:
                print(f"  {FAIL} Global Accelerator Quota (GPUS_ALL_REGIONS) is exactly 0.0.")
                print("         This blocks all GPU/TPU provisioning across all regions.")
            else:
                global_accel_ok = True
        else:
            print(f"  {PASS} GPUS_ALL_REGIONS metric not found (default unlimited or unconstrained).")
            global_accel_ok = True

    # 10. Scan for Existing Queued Resources
    print("\n🔍 [10/12] Scanning active or failed queued resources...")

    def fetch_zone_resources(zone):
        data, err = run_gcloud_json(["alpha", "compute", "tpus", "queued-resources", "list", f"--zone={zone}"])
        return zone, data, err

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(fetch_zone_resources, SCAN_ZONES))

    has_dangling = False
    for zone, qrs, err in results:
        if err and "API_disabled" not in err and "not enabled" not in err:
            print(f"  {WARN} Error scanning zone {YELLOW}{zone}{NC}: {err}")
        if qrs:
            print(f"\n  Zone: {YELLOW}{zone}{NC}")
            print("    %-30s %-20s" % ("NAME", "STATE"))
            print("    " + "-"*51)
            for qr in qrs:
                name = qr.get("name")
                short_name = name.split("/")[-1]
                state = qr.get("state", {}).get("state", "UNKNOWN")
                print("    %-30s %-20s" % (short_name, state))
                has_dangling = True
                if state == "FAILED":
                    desc, desc_err = run_gcloud_json(["alpha", "compute", "tpus", "queued-resources", "describe", short_name, f"--zone={zone}"])
                    if desc:
                        failed_data = desc.get("state", {}).get("failedData", {})
                        err_obj = failed_data.get("error", {})
                        err_code = err_obj.get("code")
                        err_msg = err_obj.get("message", "")
                        if err_code or err_msg:
                            print(f"      {RED}➔ Failure Code: {err_code} | Message: {err_msg}{NC}")
                            if "vmExternalIpAccess" in err_msg or "external IP" in err_msg:
                                print(f"        {YELLOW}💡 Hint: External IP access is blocked by organization policies. Run your command with private networking flags (e.g. without external IP).{NC}")

    if not has_dangling:
        print("  ✅ No existing queued resources detected.")

    # 10b. Scan for GCE TPU VM Instances (v6e / v5p)
    print("\n🔍 [10b/12] Scanning active Compute Engine first-class TPU VM instances...")
    gce_vms, gce_err = run_gcloud_json([
        "compute", "instances", "list",
        "--filter=machineType:(ct6e-standard* OR ct5p-hightpu* OR ct5lp-hightpu*)"
    ])
    
    has_gce = False
    if gce_err:
        print(f"  {WARN} Failed to scan Compute Engine instances: {gce_err}")
    elif gce_vms:
        print("    %-30s %-20s %-15s" % ("NAME", "ZONE", "STATUS"))
        print("    " + "-"*68)
        for vm in gce_vms:
            name = vm.get("name")
            zone_url = vm.get("zone", "")
            zone = zone_url.split("/")[-1] if zone_url else "Unknown"
            status = vm.get("status", "UNKNOWN")
            print("    %-30s %-20s %-15s" % (name, zone, status))
            has_gce = True
            
    if not has_gce and not gce_err:
        print("  ✅ No active Compute Engine TPU VM instances detected.")

    # 11. Query TPU Quotas
    print("\n🔍 [11/12] Querying TPU quotas...")

    service_quotas = []
    if alpha_ok:
        tpu_service, tpu_err = run_gcloud_json(["alpha", "services", "quota", "list", "--service=tpu.googleapis.com", f"--consumer=projects/{project_id}"], timeout=30)
        if tpu_err:
            print(f"  {WARN} Failed to query tpu.googleapis.com service quotas: {tpu_err}")
        
        compute_service, comp_err = run_gcloud_json(["alpha", "services", "quota", "list", "--service=compute.googleapis.com", f"--consumer=projects/{project_id}", "--filter=metric:(tpus_per_tpu_family OR preemptible_tpu_v5p OR preemptible_tpu_v6e OR tpu_v5p OR tpu_v6e)"], timeout=30)
        if comp_err:
            print(f"  {WARN} Failed to query compute.googleapis.com service quotas: {comp_err}")
            
        service_quotas = (tpu_service or []) + (compute_service or [])

    quota_regions = sorted(list(set(["-".join(z.split("-")[:-1]) for z in SCAN_ZONES])))
    for region in quota_regions:
        print(f"\n--- Quota Limits: {YELLOW}{region}{NC} ---")
        region_data, _ = run_gcloud_json(["compute", "regions", "describe", region])
        
        printed_header = False
        
        if region_data:
            quotas = region_data.get("quotas", [])
            tpu_quotas = [q for q in quotas if "tpu" in q.get("metric", "").lower()]
            
            if tpu_quotas:
                print("  %-45s %-10s %-10s" % ("METRIC", "LIMIT", "USAGE"))
                print("  " + "-"*67)
                printed_header = True
                for q in tpu_quotas:
                    metric_name = q.get("metric", "")
                    limit_val = q.get("limit", 0.0)
                    print("  %-45s %-10s %-10s" % (metric_name, limit_val, q.get("usage")))
                    if limit_val == 0.0 and "preemptible" not in metric_name.lower():
                        has_zero_quota = True

        matched_service_quotas = []
        for item in service_quotas:
            metric_name = item.get("metric", "")
            
            # Keep only TPU-related and queued resources service quotas
            clean_name = metric_name.replace("tpu.googleapis.com/", "").replace("compute.googleapis.com/", "").lower()
            if "tpu" not in clean_name and "queuedresources" not in clean_name:
                continue
            
            for limit_container in item.get("consumerQuotaLimits", []):
                locations = limit_container.get("supportedLocations", [])
                
                # Check if this limit applies globally or specifically to the current loop region
                if locations and not any(loc == region or loc.startswith(region + "-") for loc in locations):
                    continue
                
                buckets = limit_container.get("quotaBuckets", [])
                if not buckets:
                    continue
                
                if "tpus_per_tpu_family" in metric_name:
                    for family in ["CT3", "CT3P", "CT6E"]:
                        best_limit = "Unknown"
                        
                        # Find region-specific bucket first, fallback to global
                        for bucket in buckets:
                            dims = bucket.get("dimensions", {})
                            if dims.get("tpu_family") == family:
                                if dims.get("region") == region:
                                    best_limit = bucket.get("effectiveLimit", best_limit)
                                    break  # Region specific match is optimal
                                elif not dims.get("region") and best_limit == "Unknown":
                                    best_limit = bucket.get("effectiveLimit", best_limit)
                        
                        if best_limit == "-1": best_limit = "Unlimited"
                        elif best_limit in ["Unknown", ""]: best_limit = "0"
                            
                        clean_metric = f"tpus_per_tpu_family ({family}) (Service Quota)"
                        matched_service_quotas.append((clean_metric, best_limit))
                else:
                    eff_limit = "Unknown"
                    for bucket in buckets:
                        dims = bucket.get("dimensions", {})
                        if dims.get("region") == region or any(dims.get("zone", "").startswith(region + "-") for _ in [1]):
                            eff_limit = bucket.get("effectiveLimit", eff_limit)
                            
                    if eff_limit in ["Unknown", ""]:
                        for bucket in buckets:
                            if not bucket.get("dimensions"):
                                eff_limit = bucket.get("effectiveLimit", eff_limit)
                    
                    if eff_limit == "-1": eff_limit = "Unlimited"
                    elif eff_limit == "Unknown": eff_limit = "0"
                    
                    unit = limit_container.get("unit", "")
                    scope = ", Per-Zone" if "zone" in unit.lower() else ", Per-Region" if "region" in unit.lower() else ""
                    
                    clean_metric = clean_name + f" (Service Quota{scope})"
                    matched_service_quotas.append((clean_metric, eff_limit))

        if matched_service_quotas:
            if not printed_header:
                print("  %-45s %-10s %-10s" % ("METRIC", "LIMIT", "USAGE"))
                print("  " + "-"*67)
                printed_header = True
            for metric, limit in matched_service_quotas:
                print("  %-45s %-10s %-10s" % (metric, limit, "-"))
                if limit == "0" and "preemptible" not in metric.lower():
                    has_zero_quota = True
                
        if not printed_header:
            print("  No TPU quotas found in this region.")

    if has_zero_quota:
        print(f"\n{BLUE}💡 Tip: You have one or more TPU metrics with a limit of 0.0.")
        print("   This is completely normal for machine types you haven't explicitly requested.")
        print(f"   Just verify that the specific TPU type you want to use has a limit > 0 in your target region.{NC}")

    # 12. Check TPU Location Visibility
    print("\n🔍 [12/12] Checking TPU Location Visibility...")
    locations_data, loc_err = run_gcloud_json(["compute", "tpus", "locations", "list"])
    if loc_err:
        print(f"  {FAIL} Failed to query TPU locations list.")
        print(f"         Raw details: {loc_err}")
    elif locations_data:
        visible_zones = {loc.get("locationId") for loc in locations_data if loc.get("locationId")}
        missing_zones = [z for z in RECOMMENDED_ZONES if z not in visible_zones]
        
        if not missing_zones:
            print(f"  {PASS} All recommended zones are visible: {', '.join(RECOMMENDED_ZONES)}")
            locations_ok = True
        else:
            print(f"  {WARN} Recommended zones missing: {', '.join(missing_zones)}")
            print("         Note: If you request TPU resources in these zones, the API calls may fail with not found / permission errors.")
            locations_ok = (len(missing_zones) == 0)
    else:
        print(f"  {FAIL} No TPU locations returned by the API.")

    # SUMMARY REPORT
    print("\n========================================================")
    print("              DIAGNOSTIC SUMMARY DASHBOARD")
    print("========================================================")
    print(f"  Project ID:             {project_id}")
    print(f"  Project Number:         {project_number}")
    print(f"  Billing Account ID:     {billing_id}")
    print(f"  Google Cloud SDK Ver:   {sdk_version}")
    print("--------------------------------------------------------")

    tpu_api_badge = f"{GREEN}[OK]{NC}" if api_tpu_ok else f"{RED}[FAIL]{NC} (Action: Enable tpu.googleapis.com)"
    print(f"  API - TPU:              {tpu_api_badge}")

    comp_api_badge = f"{GREEN}[OK]{NC}" if api_compute_ok else f"{RED}[FAIL]{NC} (Action: Enable compute.googleapis.com)"
    print(f"  API - Compute:          {comp_api_badge}")

    if billing_state == "OK":
        print(f"  Billing Enabled:        {GREEN}[OK]{NC}")
    elif billing_state == "WARN":
        print(f"  Billing Enabled:        {BLUE}[INFO]{NC} (Verify manually in Console)")
    else:
        print(f"  Billing Enabled:        {RED}[FAIL]{NC} (Action: Link a billing account)")

    alpha_badge = f"{GREEN}[OK]{NC}" if alpha_ok else f"{YELLOW}[WARN]{NC} (Action: Install alpha component)"
    print(f"  gcloud Alpha Support:   {alpha_badge}")

    default_vpc_badge = f"{GREEN}[OK]{NC}" if default_vpc_ok else f"{YELLOW}[WARN]{NC} (Action: Specify network in commands)"
    print(f"  Default Network (VPC):  {default_vpc_badge}")

    tpu_sa_badge = f"{GREEN}[OK]{NC}" if tpu_sa_ok else f"{YELLOW}[WARN]{NC} (Action: Create identity)"
    print(f"  TPU Service Agent:      {tpu_sa_badge}")

    global_accel_badge = f"{GREEN}[OK]{NC}" if global_accel_ok else f"{RED}[FAIL]{NC} (Action: Request GPUS_ALL_REGIONS > 0)"
    print(f"  Global Quota (Gate):    {global_accel_badge}")

    loc_badge = f"{GREEN}[OK]{NC}" if locations_ok else f"{YELLOW}[WARN]{NC} (Action: Request zone access / contact support)"
    print(f"  Location Visibility:    {loc_badge}")

    print(f"  TPU Quotas (Standard):  {BLUE}[INFO]{NC} (Review active limits listed above)")
    print("========================================================\n")

    print(f"💾 A clean, color-stripped copy of this report has been saved to: {YELLOW}tpu_diagnostic_report.txt{NC}")
    print(f"   You can attach this file directly to your support email to help us fast-track your requests!\n")

    if not global_accel_ok:
        print(f"{YELLOW}💡 Note: If your Global Quota check failed, please review the specific error details printed above.{NC}")

if __name__ == "__main__":
    try:
        with Tee("tpu_diagnostic_report.txt"):
            main()
    except SystemExit as e:
        sys.exit(e.code)
    except KeyboardInterrupt:
        print("\n🛑 Execution cancelled by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unhandled error during execution: {e}")
        sys.exit(1)