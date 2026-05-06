from supabase_agent import SupabaseAgent

sa = SupabaseAgent()
tables = ['beliefs', 'semantic_memories', 'episodic_memories', 'user_profiles', 'self_reflections', 'agent_goals', 'products']

for t in tables:
    data = sa.read(t)
    print(f'=== {t} ({len(data)} rows) ===')
    for row in data[:3]:
        print(row)
    print()