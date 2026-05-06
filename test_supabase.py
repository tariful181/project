"""
Test Supabase Connection and Permissions
"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print("=" * 60)
print("SUPABASE CONNECTION TEST")
print("=" * 60)
print(f"\nURL: {SUPABASE_URL}")
print(f"Key exists: {bool(SUPABASE_KEY)}")
print(f"Key starts with: {SUPABASE_KEY[:20] if SUPABASE_KEY else 'None'}...")

try:
    # Test connection
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("\n✅ Successfully created Supabase client")
    
    # Try to list tables (this will test permissions)
    print("\nTesting table access...")
    tables_to_test = [
        'beliefs',
        'semantic_memories', 
        'episodic_memories',
        'user_profiles',
        'goals',
        'self_reflections',
        'agent_goals',
        'products'
    ]
    
    for table in tables_to_test:
        try:
            result = supabase.table(table).select("*").limit(1).execute()
            print(f"✅ Table '{table}': ACCESSIBLE (found {len(result.data)} rows)")
        except Exception as e:
            error_msg = str(e)
            if 'permission denied' in error_msg:
                print(f"❌ Table '{table}': PERMISSION DENIED")
            elif 'Could not find' in error_msg:
                print(f"⚠️  Table '{table}': TABLE NOT FOUND")
            else:
                print(f"❌ Table '{table}': ERROR - {error_msg[:100]}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS:")
    print("=" * 60)
    print("""
The issue is likely one of:
1. Using anon/public key instead of service_role key
2. Row Level Security (RLS) policies blocking access
3. Tables don't exist in the database
4. Schema permissions not granted

SOLUTION:
- Use the service_role key (starts with 'eyJ...') from Supabase Dashboard
- Go to: Settings > API > Project API keys
- Copy the 'service_role' key (NOT the 'anon' key)
- Update your .env file with SUPABASE_KEY=<service_role_key>
    """)

except Exception as e:
    print(f"\n❌ Failed to connect to Supabase: {e}")
    print("\nCheck your SUPABASE_URL and SUPABASE_KEY in .env file")
