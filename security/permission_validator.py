


class PermissionValidator:
    def __init__(self):
        # Define base permissions and roles
        self.permissions = {
            'read': ['user', 'admin'],
            'write': ['admin'],
            'execute': ['admin', 'system']
        }

    def is_permitted(self, action, role):
        """Check if the given role has permission for the specified action."""
        return role in self.permissions.get(action, [])

    def add_permission(self, action, role):
        """Add a new role permission for a specific action."""
        if action not in self.permissions:
            self.permissions[action] = []
        if role not in self.permissions[action]:
            self.permissions[action].append(role)

    def remove_permission(self, action, role):
        """Remove a role's permission for a specific action."""
        if action in self.permissions and role in self.permissions[action]:
            self.permissions[action].remove(role)