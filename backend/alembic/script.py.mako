"""
Project migration script template.
"""
from alembic import op
import sqlalchemy as sa

% for rev in revisions:
# Revision: ${rev}
% endfor

def upgrade():
    pass

def downgrade():
    pass
